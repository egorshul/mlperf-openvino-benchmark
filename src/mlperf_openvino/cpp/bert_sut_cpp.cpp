/**
 * BERT C++ SUT implementation for maximum throughput.
 *
 * Specialized for BERT Question Answering:
 * - 3 inputs: input_ids, attention_mask, token_type_ids (int64)
 * - 2 outputs: start_logits, end_logits (float32)
 */

#include "bert_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace mlperf_ov {

BertCppSUT::BertCppSUT(const std::string& model_path,
                       const std::string& device,
                       int num_streams,
                       const std::string& performance_hint)
    : model_path_(model_path),
      device_(device),
      num_streams_(num_streams),
      performance_hint_(performance_hint) {
}

BertCppSUT::~BertCppSUT() {
    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        response_callback_ = nullptr;
    }
}

void BertCppSUT::map_input_names() {
    const auto& inputs = model_->inputs();

    // Common BERT input name patterns
    for (const auto& input : inputs) {
        std::string name = input.get_any_name();
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (name_lower.find("input_id") != std::string::npos ||
            name_lower.find("input.1") != std::string::npos) {
            input_ids_name_ = name;
        } else if (name_lower.find("attention") != std::string::npos ||
                   name_lower.find("mask") != std::string::npos ||
                   name_lower.find("input.2") != std::string::npos) {
            attention_mask_name_ = name;
        } else if (name_lower.find("token_type") != std::string::npos ||
                   name_lower.find("segment") != std::string::npos ||
                   name_lower.find("input.3") != std::string::npos) {
            token_type_ids_name_ = name;
        }
    }

    // Fallback: use positional mapping
    if (input_ids_name_.empty() && inputs.size() >= 1) {
        input_ids_name_ = inputs[0].get_any_name();
    }
    if (attention_mask_name_.empty() && inputs.size() >= 2) {
        attention_mask_name_ = inputs[1].get_any_name();
    }
    if (token_type_ids_name_.empty() && inputs.size() >= 3) {
        token_type_ids_name_ = inputs[2].get_any_name();
    }

}

void BertCppSUT::map_output_names() {
    const auto& outputs = model_->outputs();

    if (outputs.size() == 1) {
        // Single output - assume concatenated or stacked start/end logits
        single_output_ = true;
        start_logits_name_ = outputs[0].get_any_name();
        end_logits_name_ = outputs[0].get_any_name();
    } else {
        // Multiple outputs - find start and end logits
        single_output_ = false;

        for (const auto& output : outputs) {
            std::string name = output.get_any_name();
            std::string name_lower = name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

            if (name_lower.find("start") != std::string::npos) {
                start_logits_name_ = name;
            } else if (name_lower.find("end") != std::string::npos) {
                end_logits_name_ = name;
            }
        }

        // Fallback: use positional mapping
        if (start_logits_name_.empty() && outputs.size() >= 1) {
            start_logits_name_ = outputs[0].get_any_name();
        }
        if (end_logits_name_.empty() && outputs.size() >= 2) {
            end_logits_name_ = outputs[1].get_any_name();
        }

    }
}

void BertCppSUT::load() {
    if (loaded_) {
        return;
    }


    // Read model
    model_ = core_.read_model(model_path_);

    // Map input/output names
    map_input_names();
    map_output_names();

    // Get sequence length from input shape
    const auto& inputs = model_->inputs();
    for (const auto& input : inputs) {
        if (input.get_any_name() == input_ids_name_) {
            auto shape = input.get_partial_shape();
            if (shape.rank().get_length() >= 2 && shape[1].is_static()) {
                seq_length_ = static_cast<int>(shape[1].get_length());
            }
            break;
        }
    }

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

    // Compile model
    compiled_model_ = core_.compile_model(model_, device_, properties);

    // Get optimal number of inference requests
    try {
        optimal_nireq_ = compiled_model_.get_property(ov::optimal_number_of_infer_requests);
    } catch (...) {
        optimal_nireq_ = 4;
    }


    // Create InferRequest pool (2x optimal for better pipelining)
    int num_requests = std::max(optimal_nireq_ * 2, 16);

    for (int i = 0; i < num_requests; ++i) {
        auto ctx = std::make_unique<BertInferContext>();
        ctx->request = compiled_model_.create_infer_request();
        ctx->pool_id = static_cast<size_t>(i);
        ctx->sut = this;

        // Pre-allocate input tensors (int64 for BERT)
        ov::Shape input_shape = {1, static_cast<size_t>(seq_length_)};

        ctx->input_ids_tensor = ov::Tensor(ov::element::i64, input_shape);
        ctx->attention_mask_tensor = ov::Tensor(ov::element::i64, input_shape);
        ctx->token_type_ids_tensor = ov::Tensor(ov::element::i64, input_shape);

        // Set input tensors
        ctx->request.set_tensor(input_ids_name_, ctx->input_ids_tensor);
        ctx->request.set_tensor(attention_mask_name_, ctx->attention_mask_tensor);
        ctx->request.set_tensor(token_type_ids_name_, ctx->token_type_ids_tensor);

        // Set completion callback
        BertInferContext* ctx_ptr = ctx.get();
        ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
            ctx_ptr->sut->on_inference_complete(ctx_ptr);
        });

        infer_contexts_.push_back(std::move(ctx));
        available_ids_.push(static_cast<size_t>(i));
    }


    loaded_ = true;
}

size_t BertCppSUT::get_idle_request() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    pool_cv_.wait(lock, [this]() { return !available_ids_.empty(); });

    size_t id = available_ids_.front();
    available_ids_.pop();
    return id;
}

void BertCppSUT::return_request(size_t id) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        available_ids_.push(id);
    }
    pool_cv_.notify_one();
}

void BertCppSUT::start_async(const int64_t* input_ids,
                             const int64_t* attention_mask,
                             const int64_t* token_type_ids,
                             int seq_length,
                             uint64_t query_id,
                             int sample_idx) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Get idle request from pool
    size_t id = get_idle_request();
    BertInferContext* ctx = infer_contexts_[id].get();

    // Store query info
    ctx->query_id = query_id;
    ctx->sample_idx = sample_idx;

    // Copy input data to pre-allocated tensors
    size_t copy_size = std::min(seq_length, seq_length_) * sizeof(int64_t);

    int64_t* ids_data = ctx->input_ids_tensor.data<int64_t>();
    int64_t* mask_data = ctx->attention_mask_tensor.data<int64_t>();
    int64_t* type_data = ctx->token_type_ids_tensor.data<int64_t>();

    std::memcpy(ids_data, input_ids, copy_size);
    std::memcpy(mask_data, attention_mask, copy_size);
    std::memcpy(type_data, token_type_ids, copy_size);

    // Start async inference
    pending_count_++;
    ctx->request.start_async();

    issued_count_++;
}

void BertCppSUT::on_inference_complete(BertInferContext* ctx) {
    // This runs in OpenVINO's thread - NO GIL!
    callbacks_running_++;

    size_t pool_id = ctx->pool_id;
    uint64_t query_id = ctx->query_id;
    int sample_idx = ctx->sample_idx;

    try {
        const float* start_logits_data = nullptr;
        const float* end_logits_data = nullptr;
        size_t logits_size = 0;

        if (single_output_) {
            // Single output - split into start/end
            ov::Tensor output = ctx->request.get_tensor(start_logits_name_);
            const float* data = output.data<float>();
            size_t total_size = output.get_size();

            if (output.get_shape().back() == 2) {
                // Shape: [1, seq_len, 2] - interleaved
                logits_size = total_size / 2;
                // For interleaved, we'll handle in callback
                start_logits_data = data;
                end_logits_data = data;  // Will be processed specially
            } else {
                // Shape: [1, 2*seq_len] - concatenated
                logits_size = total_size / 2;
                start_logits_data = data;
                end_logits_data = data + logits_size;
            }
        } else {
            // Two separate outputs
            ov::Tensor start_tensor = ctx->request.get_tensor(start_logits_name_);
            ov::Tensor end_tensor = ctx->request.get_tensor(end_logits_name_);

            start_logits_data = start_tensor.data<float>();
            end_logits_data = end_tensor.data<float>();
            logits_size = start_tensor.get_size();
        }

        // Store prediction if needed
        if (store_predictions_) {
            BertPrediction pred;

            if (single_output_) {
                ov::Tensor output = ctx->request.get_tensor(start_logits_name_);
                const float* data = output.data<float>();
                size_t total_size = output.get_size();

                if (output.get_shape().back() == 2) {
                    // Interleaved [1, seq_len, 2]
                    size_t seq_len = total_size / 2;
                    pred.start_logits.resize(seq_len);
                    pred.end_logits.resize(seq_len);
                    for (size_t i = 0; i < seq_len; ++i) {
                        pred.start_logits[i] = data[i * 2];
                        pred.end_logits[i] = data[i * 2 + 1];
                    }
                } else {
                    // Concatenated
                    size_t half = total_size / 2;
                    pred.start_logits.assign(data, data + half);
                    pred.end_logits.assign(data + half, data + total_size);
                }
            } else {
                pred.start_logits.assign(start_logits_data, start_logits_data + logits_size);
                pred.end_logits.assign(end_logits_data, end_logits_data + logits_size);
            }

            {
                std::lock_guard<std::mutex> lock(predictions_mutex_);
                predictions_[sample_idx] = std::move(pred);
            }
        }

        // Call response callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(query_id, start_logits_data, end_logits_data, logits_size);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[BertCppSUT] Callback error: " << e.what() << std::endl;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (response_callback_) {
                response_callback_(query_id, nullptr, nullptr, 0);
            }
        }
    }

    completed_count_++;
    pending_count_--;
    return_request(pool_id);

    callbacks_running_--;
}

void BertCppSUT::wait_all() {
    while (pending_count_.load() > 0 || callbacks_running_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BertCppSUT::reset_counters() {
    wait_all();

    issued_count_ = 0;
    completed_count_ = 0;
    callbacks_running_ = 0;
    {
        std::lock_guard<std::mutex> lock(predictions_mutex_);
        predictions_.clear();
    }
}

std::unordered_map<int, BertPrediction> BertCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void BertCppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

} // namespace mlperf_ov
