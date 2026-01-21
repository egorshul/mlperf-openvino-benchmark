/**
 * C++ SUT implementation for multi-die accelerators.
 *
 * Key design principles:
 * 1. Compile model once, then for each die
 * 2. Round-robin distribution for load balancing
 * 3. Batch handling with proper output splitting
 * 4. All callbacks in C++ without GIL
 */

#include "resnet_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>

#include <openvino/core/preprocess/pre_post_process.hpp>

namespace mlperf_ov {

ResNetMultiDieCppSUT::ResNetMultiDieCppSUT(const std::string& model_path,
                               const std::string& device_prefix,
                               int batch_size,
                               const std::unordered_map<std::string, std::string>& compile_properties,
                               bool use_nhwc_input)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      batch_size_(batch_size),
      compile_properties_(compile_properties),
      use_nhwc_input_(use_nhwc_input) {
}

ResNetMultiDieCppSUT::~ResNetMultiDieCppSUT() {
    wait_all();
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        response_callback_ = nullptr;
    }
}

std::vector<std::string> ResNetMultiDieCppSUT::discover_dies() {
    std::vector<std::string> dies;
    auto all_devices = core_.get_available_devices();

    // Pattern: PREFIX.N where N is a number
    std::regex die_pattern(device_prefix_ + R"(\.(\d+))");

    for (const auto& device : all_devices) {
        if (!device.rfind(device_prefix_, 0) == 0) {
            continue;  // Doesn't start with prefix
        }

        // Skip simulators
        if (device.find("Simulator") != std::string::npos ||
            device.find("FuncSim") != std::string::npos) {
            continue;
        }

        // Check if it matches DEVICE.N pattern
        if (std::regex_match(device, die_pattern)) {
            dies.push_back(device);
        }
    }

    // Sort by die number
    std::sort(dies.begin(), dies.end(), [this](const std::string& a, const std::string& b) {
        auto get_num = [this](const std::string& s) -> int {
            size_t pos = s.find('.');
            if (pos != std::string::npos) {
                return std::stoi(s.substr(pos + 1));
            }
            return 0;
        };
        return get_num(a) < get_num(b);
    });

    return dies;
}

ov::AnyMap ResNetMultiDieCppSUT::build_compile_properties() {
    ov::AnyMap properties;

    // Convert string properties to ov::Any
    for (const auto& [key, value] : compile_properties_) {
        // Try to parse as int
        try {
            int int_val = std::stoi(value);
            properties[key] = int_val;
            continue;
        } catch (...) {}

        // Try to parse as bool
        std::string upper_val = value;
        std::transform(upper_val.begin(), upper_val.end(), upper_val.begin(), ::toupper);
        if (upper_val == "TRUE") {
            properties[key] = true;
            continue;
        } else if (upper_val == "FALSE") {
            properties[key] = false;
            continue;
        }

        // Default: string
        properties[key] = value;
    }

    return properties;
}

void ResNetMultiDieCppSUT::load() {
    if (loaded_) {
        return;
    }

    // Discover available dies
    active_devices_ = discover_dies();
    if (active_devices_.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " dies found");
    }

    // Read model
    model_ = core_.read_model(model_path_);

    // Get input/output info
    const auto& inputs = model_->inputs();
    const auto& outputs = model_->outputs();

    if (inputs.empty() || outputs.empty()) {
        throw std::runtime_error("Model has no inputs or outputs");
    }

    input_name_ = inputs[0].get_any_name();
    input_shape_ = inputs[0].get_partial_shape().get_min_shape();
    input_type_ = inputs[0].get_element_type();

    // Find float32 output (classification logits)
    output_idx_ = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_type = outputs[i].get_element_type();
        if (out_type == ov::element::f32 || out_type == ov::element::f16) {
            output_idx_ = i;
            break;
        }
    }

    output_name_ = outputs[output_idx_].get_any_name();
    output_type_ = outputs[output_idx_].get_element_type();

    // Reshape model for batch size
    if (batch_size_ > 1 || input_shape_[0] == 0) {
        std::map<std::string, ov::PartialShape> new_shapes;
        for (const auto& input : inputs) {
            ov::PartialShape new_shape = input.get_partial_shape();
            new_shape[0] = batch_size_;
            new_shapes[input.get_any_name()] = new_shape;
        }
        model_->reshape(new_shapes);

        // Update input shape after reshape
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
    }

    // Add NHWC->NCHW transpose if using NHWC input
    if (use_nhwc_input_) {
        std::cout << "[NHWC] Original input shape: ";
        for (auto d : input_shape_) std::cout << d << " ";
        std::cout << std::endl;

        ov::preprocess::PrePostProcessor ppp(model_);
        // Input tensor is NHWC, model expects NCHW
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model_ = ppp.build();

        // Get actual input shape from the transformed model
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
        input_name_ = model_->inputs()[0].get_any_name();

        std::cout << "[NHWC] New input shape after PrePostProcessor: ";
        for (auto d : input_shape_) std::cout << d << " ";
        std::cout << "(expecting NHWC: N,H,W,C)" << std::endl;
    }

    // Build compile properties
    ov::AnyMap properties = build_compile_properties();

    // Compile model for each die
    int total_requests = 0;
    for (const auto& device_name : active_devices_) {
        auto die_ctx = std::make_unique<DieContext>();
        die_ctx->device_name = device_name;

        die_ctx->compiled_model = core_.compile_model(model_, device_name, properties);

        // Get optimal nireq
        try {
            die_ctx->optimal_nireq = die_ctx->compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (...) {
            die_ctx->optimal_nireq = 4;
        }

        // Create inference requests for this die
        // Use 8x optimal for maximum pipelining
        int num_requests = std::max(die_ctx->optimal_nireq * 8, 32);

        for (int i = 0; i < num_requests; ++i) {
            auto ctx = std::make_unique<ResNetMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = infer_contexts_.size();
            ctx->sut = this;

            // Pre-allocate input tensor
            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            // Reserve space for batch info
            ctx->query_ids.reserve(batch_size_);
            ctx->sample_indices.reserve(batch_size_);

            // Set callback
            ResNetMultiDieInferContext* ctx_ptr = ctx.get();
            ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                ctx_ptr->sut->on_inference_complete(ctx_ptr);
            });

            infer_contexts_.push_back(std::move(ctx));
            available_ids_.push(infer_contexts_.size() - 1);
        }

        total_requests += num_requests;
        die_contexts_.push_back(std::move(die_ctx));
    }

    // Calculate single output size (for splitting batched output)
    auto output_shape = model_->outputs()[output_idx_].get_partial_shape().get_min_shape();
    single_output_size_ = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {  // Skip batch dimension
        single_output_size_ *= output_shape[i];
    }

    loaded_ = true;
}

std::vector<std::string> ResNetMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

int ResNetMultiDieCppSUT::get_total_requests() const {
    return static_cast<int>(infer_contexts_.size());
}

size_t ResNetMultiDieCppSUT::get_idle_request() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    pool_cv_.wait(lock, [this]() { return !available_ids_.empty(); });

    size_t id = available_ids_.front();
    available_ids_.pop();
    return id;
}

void ResNetMultiDieCppSUT::return_request(size_t id) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        available_ids_.push(id);
    }
    pool_cv_.notify_one();
}

const std::string& ResNetMultiDieCppSUT::get_next_die() {
    size_t idx = die_index_.fetch_add(1) % active_devices_.size();
    return active_devices_[idx];
}

void ResNetMultiDieCppSUT::start_async_batch(const float* input_data,
                                        size_t input_size,
                                        const std::vector<uint64_t>& query_ids,
                                        const std::vector<int>& sample_indices,
                                        int actual_batch_size) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Get idle request
    size_t id = get_idle_request();
    ResNetMultiDieInferContext* ctx = infer_contexts_[id].get();

    // Store batch info
    ctx->query_ids = query_ids;
    ctx->sample_indices = sample_indices;
    ctx->actual_batch_size = actual_batch_size;

    // Copy input data
    float* tensor_data = ctx->input_tensor.data<float>();
    size_t tensor_bytes = ctx->input_tensor.get_byte_size();
    std::memcpy(tensor_data, input_data, std::min(input_size, tensor_bytes));

    // Start async inference
    pending_count_++;
    ctx->request.start_async();
    issued_count_ += actual_batch_size;
}

void ResNetMultiDieCppSUT::issue_queries_server(
    const std::vector<const float*>& all_input_data,
    const std::vector<size_t>& input_sizes,
    const std::vector<uint64_t>& query_ids,
    const std::vector<int>& sample_indices) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t num_samples = query_ids.size();
    size_t single_input_size = 0;
    if (!input_sizes.empty()) {
        single_input_size = input_sizes[0];
    }

    // Process all samples - dispatch to available requests
    for (size_t i = 0; i < num_samples; ++i) {
        // Get idle request (may block if all busy)
        size_t id = get_idle_request();
        ResNetMultiDieInferContext* ctx = infer_contexts_[id].get();

        // Setup for single sample
        ctx->query_ids.clear();
        ctx->query_ids.push_back(query_ids[i]);
        ctx->sample_indices.clear();
        ctx->sample_indices.push_back(sample_indices[i]);
        ctx->actual_batch_size = 1;

        // Copy input data
        float* tensor_data = ctx->input_tensor.data<float>();
        size_t copy_size = std::min(input_sizes[i], ctx->input_tensor.get_byte_size());
        std::memcpy(tensor_data, all_input_data[i], copy_size);

        // Start async inference
        pending_count_++;
        ctx->request.start_async();
        issued_count_++;
    }
}

void ResNetMultiDieCppSUT::on_inference_complete(ResNetMultiDieInferContext* ctx) {
    callbacks_running_++;

    size_t pool_id = ctx->pool_id;
    int actual_batch_size = ctx->actual_batch_size;

    try {
        // Store predictions if needed (for accuracy mode)
        if (store_predictions_) {
            ov::Tensor output_tensor = ctx->request.get_output_tensor(output_idx_);
            size_t output_bytes_per_sample = single_output_size_ * sizeof(float);

            // Handle different output types
            std::vector<float> converted_output;
            const float* output_data = nullptr;

            if (output_type_ == ov::element::f32) {
                output_data = output_tensor.data<float>();
            } else if (output_type_ == ov::element::i64) {
                const int64_t* i64_data = output_tensor.data<int64_t>();
                size_t total_elements = single_output_size_ * actual_batch_size;
                converted_output.resize(total_elements);
                for (size_t j = 0; j < total_elements; ++j) {
                    converted_output[j] = static_cast<float>(i64_data[j]);
                }
                output_data = converted_output.data();
            } else if (output_type_ == ov::element::i32) {
                const int32_t* i32_data = output_tensor.data<int32_t>();
                size_t total_elements = single_output_size_ * actual_batch_size;
                converted_output.resize(total_elements);
                for (size_t j = 0; j < total_elements; ++j) {
                    converted_output[j] = static_cast<float>(i32_data[j]);
                }
                output_data = converted_output.data();
            } else if (output_type_ == ov::element::f16) {
                converted_output.resize(single_output_size_ * actual_batch_size);
                const ov::float16* f16_data = output_tensor.data<ov::float16>();
                for (size_t j = 0; j < converted_output.size(); ++j) {
                    converted_output[j] = static_cast<float>(f16_data[j]);
                }
                output_data = converted_output.data();
            }

            if (output_data) {
                for (int i = 0; i < actual_batch_size; ++i) {
                    int sample_idx = ctx->sample_indices[i];
                    const float* sample_output = output_data + (i * single_output_size_);
                    std::vector<float> prediction(sample_output, sample_output + single_output_size_);
                    {
                        std::lock_guard<std::mutex> lock(predictions_mutex_);
                        predictions_[sample_idx] = std::move(prediction);
                    }
                }
            }
        }

        // Call response callback immediately (Server mode requires low latency!)
        {
            std::lock_guard<std::mutex> cb_lock(callback_mutex_);
            if (batch_response_callback_) {
                // Send immediately - don't batch for Server mode latency
                batch_response_callback_(ctx->query_ids);
            } else if (response_callback_) {
                for (int i = 0; i < actual_batch_size; ++i) {
                    response_callback_(ctx->query_ids[i], nullptr, 0);
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ResNetMultiDieCppSUT] Callback error on " << ctx->die_name << ": " << e.what() << std::endl;

        // Still send responses immediately to avoid hang
        try {
            std::lock_guard<std::mutex> cb_lock(callback_mutex_);
            if (batch_response_callback_) {
                batch_response_callback_(ctx->query_ids);
            } else if (response_callback_) {
                for (int i = 0; i < actual_batch_size; ++i) {
                    response_callback_(ctx->query_ids[i], nullptr, 0);
                }
            }
        } catch (...) {
            // Ignore callback errors in error handler
        }
    }

    completed_count_ += actual_batch_size;
    pending_count_--;
    return_request(pool_id);
    callbacks_running_--;
}

void ResNetMultiDieCppSUT::flush_pending_responses() {
    std::vector<uint64_t> to_flush;
    {
        std::lock_guard<std::mutex> lock(pending_responses_mutex_);
        if (!pending_responses_.empty()) {
            to_flush = std::move(pending_responses_);
            pending_responses_.clear();
        }
    }

    if (!to_flush.empty()) {
        std::lock_guard<std::mutex> cb_lock(callback_mutex_);
        if (batch_response_callback_) {
            batch_response_callback_(to_flush);
        } else if (response_callback_) {
            for (auto qid : to_flush) {
                response_callback_(qid, nullptr, 0);
            }
        }
    }
}

void ResNetMultiDieCppSUT::wait_all() {
    while (pending_count_.load() > 0 || callbacks_running_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Flush any remaining responses
    flush_pending_responses();
}

void ResNetMultiDieCppSUT::reset_counters() {
    wait_all();

    issued_count_ = 0;
    completed_count_ = 0;
    die_index_ = 0;
    // Note: Don't clear predictions here - they accumulate across LoadGen calls
}

std::unordered_map<int, std::vector<float>> ResNetMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void ResNetMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void ResNetMultiDieCppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

void ResNetMultiDieCppSUT::set_batch_response_callback(BatchResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_response_callback_ = callback;
}

} // namespace mlperf_ov
