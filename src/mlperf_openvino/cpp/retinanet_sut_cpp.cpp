/**
 * RetinaNet C++ SUT implementation for maximum throughput.
 */

#include "retinanet_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace mlperf_ov {

RetinaNetCppSUT::RetinaNetCppSUT(const std::string& model_path,
                                   const std::string& device,
                                   int num_streams,
                                   const std::string& performance_hint)
    : model_path_(model_path),
      device_(device),
      num_streams_(num_streams),
      performance_hint_(performance_hint) {
}

RetinaNetCppSUT::~RetinaNetCppSUT() {
    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        response_callback_ = nullptr;
    }
}

void RetinaNetCppSUT::map_output_names() {
    const auto& outputs = model_->outputs();

    // Reset names
    boxes_name_.clear();
    scores_name_.clear();
    labels_name_.clear();

    // Find output names by examining names and shapes
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        std::string name = output.get_any_name();
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        auto shape = output.get_partial_shape();

        if (name_lower.find("box") != std::string::npos ||
            name_lower.find("bbox") != std::string::npos) {
            boxes_name_ = name;
            boxes_idx_ = static_cast<int>(i);
        } else if (name_lower.find("score") != std::string::npos ||
                   name_lower.find("conf") != std::string::npos) {
            scores_name_ = name;
            scores_idx_ = static_cast<int>(i);
        } else if (name_lower.find("label") != std::string::npos ||
                   name_lower.find("class") != std::string::npos) {
            labels_name_ = name;
            labels_idx_ = static_cast<int>(i);
        }
    }

    // Fallback: use positional mapping
    // MLPerf RetinaNet ONNX model outputs: [0]=boxes, [1]=scores, [2]=labels
    if (boxes_name_.empty() && outputs.size() >= 1) {
        boxes_name_ = outputs[0].get_any_name();
        boxes_idx_ = 0;
    }
    if (scores_name_.empty() && outputs.size() >= 2) {
        scores_name_ = outputs[1].get_any_name();
        scores_idx_ = 1;
    }
    if (labels_name_.empty() && outputs.size() >= 3) {
        labels_name_ = outputs[2].get_any_name();
        labels_idx_ = 2;
    }

    std::cout << "[RetinaNetCppSUT] Output mapping:" << std::endl;
    std::cout << "  boxes[" << boxes_idx_ << "]: " << boxes_name_ << std::endl;
    std::cout << "  scores[" << scores_idx_ << "]: " << scores_name_ << std::endl;
    std::cout << "  labels[" << labels_idx_ << "]: " << labels_name_ << std::endl;
}

void RetinaNetCppSUT::load() {
    if (loaded_) {
        return;
    }

    std::cout << "[RetinaNetCppSUT] Loading model: " << model_path_ << std::endl;

    model_ = core_.read_model(model_path_);

    // Get input info
    const auto& inputs = model_->inputs();
    if (inputs.empty()) {
        throw std::runtime_error("Model has no inputs");
    }

    input_name_ = inputs[0].get_any_name();
    input_shape_ = inputs[0].get_partial_shape().get_min_shape();

    // Handle dynamic batch
    if (input_shape_.size() > 0 && input_shape_[0] == 0) {
        input_shape_[0] = 1;
    }

    std::cout << "[RetinaNetCppSUT] Input: " << input_name_ << ", shape: " << input_shape_ << std::endl;

    // Map output names
    map_output_names();

    // Build compile properties
    ov::AnyMap properties;

    if (performance_hint_ == "THROUGHPUT") {
        properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    } else if (performance_hint_ == "LATENCY") {
        properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    }

    if (num_streams_ > 0) {
        properties[ov::hint::num_requests.name()] = num_streams_;
    }

    if (device_ == "CPU") {
        properties[ov::hint::enable_cpu_pinning.name()] = true;
    }

    std::cout << "[RetinaNetCppSUT] Compiling model for device: " << device_ << std::endl;
    compiled_model_ = core_.compile_model(model_, device_, properties);

    try {
        optimal_nireq_ = compiled_model_.get_property(ov::optimal_number_of_infer_requests);
    } catch (...) {
        optimal_nireq_ = 4;
    }

    std::cout << "[RetinaNetCppSUT] Optimal requests: " << optimal_nireq_ << std::endl;

    // Create InferRequest pool
    int num_requests = std::max(optimal_nireq_ * 2, 16);

    for (int i = 0; i < num_requests; ++i) {
        auto ctx = std::make_unique<RetinaNetInferContext>();
        ctx->request = compiled_model_.create_infer_request();
        ctx->pool_id = static_cast<size_t>(i);
        ctx->sut = this;

        // Pre-allocate input tensor
        ctx->input_tensor = ov::Tensor(ov::element::f32, input_shape_);
        ctx->request.set_input_tensor(ctx->input_tensor);

        // Set completion callback
        RetinaNetInferContext* ctx_ptr = ctx.get();
        ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
            ctx_ptr->sut->on_inference_complete(ctx_ptr);
        });

        infer_contexts_.push_back(std::move(ctx));
        available_ids_.push(static_cast<size_t>(i));
    }

    std::cout << "[RetinaNetCppSUT] Created " << num_requests << " inference requests" << std::endl;

    loaded_ = true;
    std::cout << "[RetinaNetCppSUT] Model loaded successfully" << std::endl;
}

std::vector<size_t> RetinaNetCppSUT::get_input_shape() const {
    return std::vector<size_t>(input_shape_.begin(), input_shape_.end());
}

size_t RetinaNetCppSUT::get_idle_request() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    pool_cv_.wait(lock, [this]() { return !available_ids_.empty(); });

    size_t id = available_ids_.front();
    available_ids_.pop();
    return id;
}

void RetinaNetCppSUT::return_request(size_t id) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        available_ids_.push(id);
    }
    pool_cv_.notify_one();
}

void RetinaNetCppSUT::start_async(const float* input_data,
                                   size_t input_size,
                                   uint64_t query_id,
                                   int sample_idx) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t id = get_idle_request();
    RetinaNetInferContext* ctx = infer_contexts_[id].get();

    ctx->query_id = query_id;
    ctx->sample_idx = sample_idx;

    // Copy input data
    float* tensor_data = ctx->input_tensor.data<float>();
    size_t tensor_size = ctx->input_tensor.get_size();
    std::memcpy(tensor_data, input_data, std::min(input_size, tensor_size) * sizeof(float));

    pending_count_++;
    ctx->request.start_async();

    issued_count_++;
}

void RetinaNetCppSUT::on_inference_complete(RetinaNetInferContext* ctx) {
    callbacks_running_++;

    size_t pool_id = ctx->pool_id;
    uint64_t query_id = ctx->query_id;
    int sample_idx = ctx->sample_idx;

    try {
        // Get output tensors
        ov::Tensor boxes_tensor = ctx->request.get_output_tensor(boxes_idx_);
        ov::Tensor scores_tensor = ctx->request.get_output_tensor(scores_idx_);

        const float* boxes_data = boxes_tensor.data<float>();
        const float* scores_data = scores_tensor.data<float>();
        size_t boxes_size = boxes_tensor.get_size();
        size_t scores_size = scores_tensor.get_size();

        // Labels handling - model outputs int64_t, we convert to float for API compatibility
        std::vector<float> labels_float;
        const float* labels_data = nullptr;
        size_t labels_size = 0;

        if (labels_idx_ < static_cast<int>(ctx->request.get_compiled_model().outputs().size())) {
            try {
                ov::Tensor labels_tensor = ctx->request.get_output_tensor(labels_idx_);
                labels_size = labels_tensor.get_size();

                // Check element type and convert if needed
                auto elem_type = labels_tensor.get_element_type();
                if (elem_type == ov::element::i64) {
                    // Model outputs int64_t labels - convert to float
                    const int64_t* labels_int64 = labels_tensor.data<int64_t>();
                    labels_float.resize(labels_size);
                    for (size_t i = 0; i < labels_size; ++i) {
                        labels_float[i] = static_cast<float>(labels_int64[i]);
                    }
                    labels_data = labels_float.data();
                } else if (elem_type == ov::element::i32) {
                    // Model outputs int32_t labels - convert to float
                    const int32_t* labels_int32 = labels_tensor.data<int32_t>();
                    labels_float.resize(labels_size);
                    for (size_t i = 0; i < labels_size; ++i) {
                        labels_float[i] = static_cast<float>(labels_int32[i]);
                    }
                    labels_data = labels_float.data();
                } else {
                    // Assume float32
                    labels_data = labels_tensor.data<float>();
                }
            } catch (...) {
                // Labels output not available
            }
        }

        // Store prediction if needed
        if (store_predictions_) {
            RetinaNetPrediction pred;
            pred.boxes.assign(boxes_data, boxes_data + boxes_size);
            pred.scores.assign(scores_data, scores_data + scores_size);
            if (labels_data && labels_size > 0) {
                pred.labels.assign(labels_data, labels_data + labels_size);
            }
            pred.num_detections = static_cast<int>(scores_size);

            {
                std::lock_guard<std::mutex> lock(predictions_mutex_);
                predictions_[sample_idx] = std::move(pred);
            }
        }

        // Call response callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(query_id,
                                  boxes_data, boxes_size,
                                  scores_data, scores_size,
                                  labels_data, labels_size);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[RetinaNetCppSUT] Callback error: " << e.what() << std::endl;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(query_id, nullptr, 0, nullptr, 0, nullptr, 0);
            }
        }
    }

    completed_count_++;
    pending_count_--;
    return_request(pool_id);

    callbacks_running_--;
}

void RetinaNetCppSUT::wait_all() {
    while (pending_count_.load() > 0 || callbacks_running_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RetinaNetCppSUT::reset_counters() {
    wait_all();

    issued_count_ = 0;
    completed_count_ = 0;
    callbacks_running_ = 0;
    {
        std::lock_guard<std::mutex> lock(predictions_mutex_);
        predictions_.clear();
    }
}

std::unordered_map<int, RetinaNetPrediction> RetinaNetCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void RetinaNetCppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

} // namespace mlperf_ov
