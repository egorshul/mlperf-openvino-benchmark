/**
 * C++ RetinaNet SUT implementation for multi-die accelerators.
 */

#include "retinanet_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>

#include <openvino/core/preprocess/pre_post_process.hpp>

namespace mlperf_ov {

RetinaNetMultiDieCppSUT::RetinaNetMultiDieCppSUT(
    const std::string& model_path,
    const std::string& device_prefix,
    const std::unordered_map<std::string, std::string>& compile_properties,
    bool use_nhwc_input)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      compile_properties_(compile_properties),
      use_nhwc_input_(use_nhwc_input) {
}

RetinaNetMultiDieCppSUT::~RetinaNetMultiDieCppSUT() {
    wait_all();
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        response_callback_ = nullptr;
    }
}

std::vector<std::string> RetinaNetMultiDieCppSUT::discover_dies() {
    std::vector<std::string> dies;
    auto all_devices = core_.get_available_devices();

    std::regex die_pattern(device_prefix_ + R"(\.(\d+))");

    for (const auto& device : all_devices) {
        if (device.rfind(device_prefix_, 0) != 0) {
            continue;
        }

        std::smatch match;
        if (std::regex_match(device, match, die_pattern)) {
            // Check it's not a simulator
            try {
                std::string full_name = core_.get_property(device, ov::device::full_name);
                std::string lower_name = full_name;
                std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

                if (lower_name.find("simulator") != std::string::npos ||
                    lower_name.find("funcsimulator") != std::string::npos) {
                    continue;
                }
            } catch (...) {
                // Ignore errors
            }

            dies.push_back(device);
        }
    }

    std::sort(dies.begin(), dies.end());
    return dies;
}

ov::AnyMap RetinaNetMultiDieCppSUT::build_compile_properties() {
    ov::AnyMap properties;

    for (const auto& [key, value] : compile_properties_) {
        properties[key] = value;
    }

    return properties;
}

void RetinaNetMultiDieCppSUT::map_output_names() {
    const auto& outputs = model_->outputs();

    // Try to identify outputs by name patterns
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string name = outputs[i].get_any_name();
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        if (lower_name.find("box") != std::string::npos) {
            boxes_idx_ = i;
            boxes_name_ = name;
        } else if (lower_name.find("score") != std::string::npos) {
            scores_idx_ = i;
            scores_name_ = name;
        } else if (lower_name.find("label") != std::string::npos ||
                   lower_name.find("class") != std::string::npos) {
            labels_idx_ = i;
            labels_name_ = name;
        }
    }

    // Fallback to positional if names not found
    if (boxes_name_.empty() && outputs.size() > 0) {
        boxes_idx_ = 0;
        boxes_name_ = outputs[0].get_any_name();
    }
    if (scores_name_.empty() && outputs.size() > 1) {
        scores_idx_ = 1;
        scores_name_ = outputs[1].get_any_name();
    }
    if (labels_name_.empty() && outputs.size() > 2) {
        labels_idx_ = 2;
        labels_name_ = outputs[2].get_any_name();
    }
}

void RetinaNetMultiDieCppSUT::load() {
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

    // Map output names
    map_output_names();

    // Add NHWC->NCHW transpose if using NHWC input
    if (use_nhwc_input_) {
        std::cout << "[NHWC] RetinaNet original input shape: ";
        for (auto d : input_shape_) std::cout << d << " ";
        std::cout << std::endl;

        ov::preprocess::PrePostProcessor ppp(model_);
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model_ = ppp.build();

        // Get actual input shape from the transformed model
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
        input_name_ = model_->inputs()[0].get_any_name();

        std::cout << "[NHWC] RetinaNet new input shape: ";
        for (auto d : input_shape_) std::cout << d << " ";
        std::cout << std::endl;
    }

    // Build compile properties
    ov::AnyMap properties = build_compile_properties();

    // Compile model for each die
    int total_requests = 0;
    for (const auto& device_name : active_devices_) {
        auto die_ctx = std::make_unique<RetinaNetDieContext>();
        die_ctx->device_name = device_name;

        die_ctx->compiled_model = core_.compile_model(model_, device_name, properties);

        // Get optimal nireq
        try {
            die_ctx->optimal_nireq = die_ctx->compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (...) {
            die_ctx->optimal_nireq = 4;
        }

        // Create inference requests for this die
        int num_requests = std::max(die_ctx->optimal_nireq * 8, 32);

        for (int i = 0; i < num_requests; ++i) {
            auto ctx = std::make_unique<RetinaNetMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = infer_contexts_.size();
            ctx->sut = this;

            // Pre-allocate input tensor
            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            // Set callback
            RetinaNetMultiDieInferContext* ctx_ptr = ctx.get();
            ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                ctx_ptr->sut->on_inference_complete(ctx_ptr);
            });

            infer_contexts_.push_back(std::move(ctx));
            available_ids_.push(infer_contexts_.size() - 1);
        }

        total_requests += num_requests;
        die_contexts_.push_back(std::move(die_ctx));
    }

    loaded_ = true;
}

std::vector<std::string> RetinaNetMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

size_t RetinaNetMultiDieCppSUT::get_idle_request() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    pool_cv_.wait(lock, [this] { return !available_ids_.empty(); });
    size_t id = available_ids_.front();
    available_ids_.pop();
    return id;
}

void RetinaNetMultiDieCppSUT::return_request(size_t id) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        available_ids_.push(id);
    }
    pool_cv_.notify_one();
}

void RetinaNetMultiDieCppSUT::start_async(const float* input_data,
                                          size_t input_size,
                                          uint64_t query_id,
                                          int sample_idx) {
    size_t ctx_id = get_idle_request();
    auto& ctx = infer_contexts_[ctx_id];

    ctx->query_id = query_id;
    ctx->sample_idx = sample_idx;

    // Copy input data
    float* tensor_data = ctx->input_tensor.data<float>();
    std::memcpy(tensor_data, input_data, input_size * sizeof(float));

    pending_count_++;
    issued_count_++;

    ctx->request.start_async();
}

void RetinaNetMultiDieCppSUT::on_inference_complete(RetinaNetMultiDieInferContext* ctx) {
    callbacks_running_++;

    try {
        // Get outputs
        ov::Tensor boxes_tensor = ctx->request.get_output_tensor(boxes_idx_);
        ov::Tensor scores_tensor = ctx->request.get_output_tensor(scores_idx_);
        ov::Tensor labels_tensor = ctx->request.get_output_tensor(labels_idx_);

        const float* boxes_data = boxes_tensor.data<float>();
        const float* scores_data = scores_tensor.data<float>();
        const float* labels_data = labels_tensor.data<float>();

        size_t boxes_size = boxes_tensor.get_size();
        size_t scores_size = scores_tensor.get_size();
        size_t labels_size = labels_tensor.get_size();

        // Store prediction if needed
        if (store_predictions_) {
            RetinaNetMultiDiePrediction pred;
            pred.boxes.assign(boxes_data, boxes_data + boxes_size);
            pred.scores.assign(scores_data, scores_data + scores_size);
            pred.labels.assign(labels_data, labels_data + labels_size);
            pred.num_detections = static_cast<int>(scores_size);

            std::lock_guard<std::mutex> lock(predictions_mutex_);
            predictions_[ctx->sample_idx] = std::move(pred);
        }

        // Call response callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(ctx->query_id,
                                   boxes_data, boxes_size,
                                   scores_data, scores_size,
                                   labels_data, labels_size);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[RetinaNetMultiDieCppSUT] Inference error: " << e.what() << std::endl;
    }

    completed_count_++;
    pending_count_--;
    callbacks_running_--;

    return_request(ctx->pool_id);
}

void RetinaNetMultiDieCppSUT::wait_all() {
    while (pending_count_.load() > 0 || callbacks_running_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RetinaNetMultiDieCppSUT::reset_counters() {
    wait_all();

    issued_count_ = 0;
    completed_count_ = 0;
    die_index_ = 0;
}

std::unordered_map<int, RetinaNetMultiDiePrediction> RetinaNetMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void RetinaNetMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void RetinaNetMultiDieCppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

} // namespace mlperf_ov
