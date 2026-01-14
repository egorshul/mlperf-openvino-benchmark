/**
 * C++ SUT implementation for maximum throughput.
 *
 * Key design principles (from Intel/NVIDIA MLPerf submissions):
 * 1. All inference callbacks execute in C++ without GIL
 * 2. Direct memory access to avoid copies
 * 3. Lock-free counters for statistics
 * 4. Minimal allocations in hot path
 */

#include "sut_cpp.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace mlperf_ov {

CppSUT::CppSUT(const std::string& model_path,
               const std::string& device,
               int num_streams,
               const std::string& performance_hint)
    : model_path_(model_path),
      device_(device),
      num_streams_(num_streams),
      performance_hint_(performance_hint),
      optimal_nireq_(1),
      loaded_(false),
      issued_count_(0),
      completed_count_(0),
      pending_count_(0),
      store_predictions_(false),
      response_callback_(nullptr) {
}

CppSUT::~CppSUT() {
    wait_all();
}

void CppSUT::load() {
    if (loaded_) {
        return;
    }

    std::cout << "[CppSUT] Loading model: " << model_path_ << std::endl;

    // Read model
    model_ = core_.read_model(model_path_);

    // Get input/output info
    const auto& inputs = model_->inputs();
    const auto& outputs = model_->outputs();

    if (inputs.empty() || outputs.empty()) {
        throw std::runtime_error("Model has no inputs or outputs");
    }

    input_name_ = inputs[0].get_any_name();
    output_name_ = outputs[0].get_any_name();
    input_shape_ = inputs[0].get_partial_shape().get_min_shape();
    input_type_ = inputs[0].get_element_type();

    std::cout << "[CppSUT] Input: " << input_name_ << ", shape: " << input_shape_ << std::endl;
    std::cout << "[CppSUT] Output: " << output_name_ << std::endl;

    // Build compile properties
    ov::AnyMap properties;

    // Performance hint
    if (performance_hint_ == "THROUGHPUT") {
        properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    } else if (performance_hint_ == "LATENCY") {
        properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    }

    // Number of streams
    if (num_streams_ > 0) {
        properties[ov::hint::num_requests.name()] = num_streams_;
    }

    // CPU-specific optimizations
    if (device_ == "CPU") {
        properties[ov::hint::enable_cpu_pinning.name()] = true;
    }

    // Compile model
    std::cout << "[CppSUT] Compiling model for device: " << device_ << std::endl;
    compiled_model_ = core_.compile_model(model_, device_, properties);

    // Get optimal number of inference requests
    try {
        optimal_nireq_ = compiled_model_.get_property(ov::optimal_number_of_infer_requests);
    } catch (...) {
        optimal_nireq_ = 4;
    }

    std::cout << "[CppSUT] Optimal number of inference requests: " << optimal_nireq_ << std::endl;

    // Create InferRequest pool with 2x optimal requests for better pipelining
    int num_requests = std::max(optimal_nireq_ * 2, 16);

    for (int i = 0; i < num_requests; ++i) {
        auto ctx = std::make_unique<InferContext>();
        ctx->request = compiled_model_.create_infer_request();
        ctx->pool_id = static_cast<size_t>(i);
        ctx->sut = this;

        // Set completion callback - this runs in OpenVINO thread WITHOUT GIL!
        InferContext* ctx_ptr = ctx.get();
        ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
            ctx_ptr->sut->on_inference_complete(ctx_ptr);
        });

        infer_contexts_.push_back(std::move(ctx));
        available_ids_.push(static_cast<size_t>(i));
    }

    std::cout << "[CppSUT] Created " << num_requests << " inference requests" << std::endl;

    loaded_ = true;
    std::cout << "[CppSUT] Model loaded successfully" << std::endl;
}

std::string CppSUT::get_input_name() const {
    return input_name_;
}

std::string CppSUT::get_output_name() const {
    return output_name_;
}

size_t CppSUT::get_idle_request() {
    std::unique_lock<std::mutex> lock(pool_mutex_);

    // Wait for an available request
    pool_cv_.wait(lock, [this]() { return !available_ids_.empty(); });

    size_t id = available_ids_.front();
    available_ids_.pop();
    return id;
}

void CppSUT::return_request(size_t id) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        available_ids_.push(id);
    }
    pool_cv_.notify_one();
}

void CppSUT::start_async(const float* input_data,
                         size_t input_size,
                         uint64_t query_id,
                         int sample_idx) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Get idle request from pool (blocks if none available)
    size_t id = get_idle_request();
    InferContext* ctx = infer_contexts_[id].get();

    // Store query info in context
    ctx->query_id = query_id;
    ctx->sample_idx = sample_idx;

    // Copy input data into the request's pre-allocated tensor
    // This is safer than using external pointer which may be freed by Python
    ov::Tensor input_tensor = ctx->request.get_input_tensor();
    float* tensor_data = input_tensor.data<float>();
    size_t tensor_size = input_tensor.get_size();
    std::memcpy(tensor_data, input_data, tensor_size * sizeof(float));

    // Start async inference
    pending_count_++;
    ctx->request.start_async();

    issued_count_++;
}

void CppSUT::on_inference_complete(InferContext* ctx) {
    // This runs in OpenVINO's internal thread - NO GIL!

    try {
        // Get softmax output tensor (index 1 for ResNet50 with ArgMax+Softmax outputs)
        // Output 0 is ArgMax (i64), Output 1 is softmax_tensor (f32)
        size_t output_idx = ctx->request.get_compiled_model().outputs().size() > 1 ? 1 : 0;
        ov::Tensor output_tensor = ctx->request.get_output_tensor(output_idx);
        const float* output_data = output_tensor.data<float>();
        size_t output_size = output_tensor.get_byte_size();

        // Store prediction if needed (for accuracy mode)
        if (store_predictions_) {
            std::vector<float> prediction(output_data,
                                          output_data + output_tensor.get_size());
            {
                std::lock_guard<std::mutex> lock(predictions_mutex_);
                predictions_[ctx->sample_idx] = std::move(prediction);
            }
        }

        // Call response callback to notify LoadGen
        // This callback is set from Python and calls QuerySamplesComplete
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(ctx->query_id, output_data, output_size);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[CppSUT] Callback error: " << e.what() << std::endl;
        // Still call response callback with null to avoid LoadGen hang
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(ctx->query_id, nullptr, 0);
            }
        }
    }

    completed_count_++;
    pending_count_--;

    // Return request to pool
    return_request(ctx->pool_id);
}

void CppSUT::wait_all() {
    // Wait for all pending requests to complete
    while (pending_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void CppSUT::reset_counters() {
    issued_count_ = 0;
    completed_count_ = 0;
    {
        std::lock_guard<std::mutex> lock(predictions_mutex_);
        predictions_.clear();
    }
}

std::unordered_map<int, std::vector<float>> CppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(predictions_mutex_));
    return predictions_;
}

void CppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

} // namespace mlperf_ov
