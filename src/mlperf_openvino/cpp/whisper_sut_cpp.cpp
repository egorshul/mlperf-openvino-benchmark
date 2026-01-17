/**
 * Whisper C++ SUT implementation for maximum throughput.
 *
 * Encoder-decoder architecture with KV-cache for efficient decoding.
 */

#include "whisper_sut_cpp.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace mlperf_ov {

WhisperCppSUT::WhisperCppSUT(const std::string& encoder_path,
                             const std::string& decoder_path,
                             const std::string& device,
                             int num_streams,
                             int max_new_tokens)
    : encoder_path_(encoder_path),
      decoder_path_(decoder_path),
      device_(device),
      num_streams_(num_streams),
      max_new_tokens_(max_new_tokens) {
}

WhisperCppSUT::~WhisperCppSUT() {
    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        response_callback_ = nullptr;
    }
}

void WhisperCppSUT::map_encoder_names() {
    const auto& inputs = encoder_model_->inputs();
    const auto& outputs = encoder_model_->outputs();

    // Encoder typically has one input (mel spectrogram)
    if (!inputs.empty()) {
        encoder_input_name_ = inputs[0].get_any_name();

        // Get mel dimensions from input shape
        auto shape = inputs[0].get_partial_shape();
        if (shape.rank().get_length() >= 3) {
            if (shape[1].is_static()) {
                n_mels_ = static_cast<int>(shape[1].get_length());
            }
            if (shape[2].is_static()) {
                n_frames_ = static_cast<int>(shape[2].get_length());
            }
        }
    }

    // Encoder output (hidden states)
    if (!outputs.empty()) {
        encoder_output_name_ = outputs[0].get_any_name();

        auto shape = outputs[0].get_partial_shape();
        if (shape.rank().get_length() >= 3) {
            if (shape[1].is_static()) {
                encoder_seq_len_ = static_cast<size_t>(shape[1].get_length());
            }
            if (shape[2].is_static()) {
                encoder_hidden_size_ = static_cast<size_t>(shape[2].get_length());
            }
        }
    }
}

void WhisperCppSUT::map_decoder_names() {
    const auto& inputs = decoder_model_->inputs();
    const auto& outputs = decoder_model_->outputs();

    // Maps to track KV-cache inputs/outputs by layer
    std::map<int, KVCacheEntry> kv_map;

    // Find decoder inputs by name patterns
    for (const auto& input : inputs) {
        std::string name = input.get_any_name();
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (name_lower.find("input_id") != std::string::npos ||
            name_lower.find("decoder_input") != std::string::npos) {
            decoder_input_ids_name_ = name;
        } else if (name_lower.find("encoder_hidden") != std::string::npos ||
                   name_lower.find("encoder_output") != std::string::npos) {
            decoder_encoder_hidden_name_ = name;
        } else if (name_lower.find("beam_idx") != std::string::npos) {
            // beam_idx input for KV-cache reordering in beam search
            decoder_beam_idx_name_ = name;
            has_beam_idx_ = true;
        } else if (name_lower.find("past_key_value") != std::string::npos ||
                   name_lower.find("past_key_values") != std::string::npos ||
                   name_lower.find("pkv") != std::string::npos) {
            // KV-cache input - parse layer index
            has_kv_cache_ = true;

            // Try to extract layer number: past_key_values.0.key, pkv_0_key, etc.
            std::regex layer_regex(R"((\d+))");
            std::smatch match;
            int layer = -1;
            if (std::regex_search(name, match, layer_regex)) {
                layer = std::stoi(match[1]);
            }

            if (layer >= 0) {
                if (kv_map.find(layer) == kv_map.end()) {
                    kv_map[layer] = KVCacheEntry{};
                }

                if (name_lower.find("key") != std::string::npos) {
                    kv_map[layer].key_input_name = name;
                    kv_map[layer].key_shape = input.get_partial_shape().get_max_shape();
                } else if (name_lower.find("value") != std::string::npos) {
                    kv_map[layer].value_input_name = name;
                    kv_map[layer].value_shape = input.get_partial_shape().get_max_shape();
                }
            }
        }
    }

    // Find KV-cache outputs (present_key_values)
    for (const auto& output : outputs) {
        std::string name = output.get_any_name();
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (name_lower.find("logits") != std::string::npos ||
            name_lower.find("output") != std::string::npos) {
            if (decoder_output_name_.empty()) {
                decoder_output_name_ = name;
            }
        } else if (name_lower.find("present") != std::string::npos ||
                   name_lower.find("key_values") != std::string::npos) {
            // KV-cache output
            std::regex layer_regex(R"((\d+))");
            std::smatch match;
            int layer = -1;
            if (std::regex_search(name, match, layer_regex)) {
                layer = std::stoi(match[1]);
            }

            if (layer >= 0 && kv_map.find(layer) != kv_map.end()) {
                if (name_lower.find("key") != std::string::npos) {
                    kv_map[layer].key_output_name = name;
                } else if (name_lower.find("value") != std::string::npos) {
                    kv_map[layer].value_output_name = name;
                }
            }
        }
    }

    // Fallback: positional mapping for basic inputs
    if (decoder_input_ids_name_.empty() && inputs.size() >= 1) {
        decoder_input_ids_name_ = inputs[0].get_any_name();
    }
    if (decoder_encoder_hidden_name_.empty() && inputs.size() >= 2) {
        decoder_encoder_hidden_name_ = inputs[1].get_any_name();
    }

    // Decoder output (logits)
    if (decoder_output_name_.empty() && !outputs.empty()) {
        decoder_output_name_ = outputs[0].get_any_name();
    }

    // Build ordered KV-cache entries
    if (has_kv_cache_) {
        num_layers_ = kv_map.size();
        kv_cache_entries_.reserve(num_layers_);

        for (int i = 0; i < static_cast<int>(num_layers_); ++i) {
            if (kv_map.find(i) != kv_map.end()) {
                kv_cache_entries_.push_back(kv_map[i]);

                // Extract num_heads and head_dim from shape
                // Shape is typically [batch, num_heads, seq_len, head_dim]
                if (!kv_map[i].key_shape.empty() && kv_map[i].key_shape.size() >= 4) {
                    num_heads_ = kv_map[i].key_shape[1];
                    head_dim_ = kv_map[i].key_shape[3];
                }
            }
        }
    }
}

void WhisperCppSUT::init_kv_cache() {
    if (!has_kv_cache_) return;

    // Initialize all KV-cache inputs with zeros (seq_len = 0 for first step)
    for (const auto& entry : kv_cache_entries_) {
        // Create zero tensors with seq_len = 0
        // Shape: [batch=1, num_heads, seq_len=0, head_dim]
        ov::Shape key_shape = {1, num_heads_, 0, head_dim_};
        ov::Shape value_shape = {1, num_heads_, 0, head_dim_};

        ov::Tensor key_tensor(ov::element::f32, key_shape);
        ov::Tensor value_tensor(ov::element::f32, value_shape);

        decoder_request_.set_tensor(entry.key_input_name, key_tensor);
        decoder_request_.set_tensor(entry.value_input_name, value_tensor);
    }
}

void WhisperCppSUT::update_kv_cache() {
    if (!has_kv_cache_) return;

    // Copy present KV-cache outputs to past KV-cache inputs
    for (const auto& entry : kv_cache_entries_) {
        if (!entry.key_output_name.empty() && !entry.key_input_name.empty()) {
            ov::Tensor present_key = decoder_request_.get_tensor(entry.key_output_name);
            // Create a copy
            ov::Tensor past_key(present_key.get_element_type(), present_key.get_shape());
            std::memcpy(past_key.data(), present_key.data(), present_key.get_byte_size());
            decoder_request_.set_tensor(entry.key_input_name, past_key);
        }

        if (!entry.value_output_name.empty() && !entry.value_input_name.empty()) {
            ov::Tensor present_value = decoder_request_.get_tensor(entry.value_output_name);
            // Create a copy
            ov::Tensor past_value(present_value.get_element_type(), present_value.get_shape());
            std::memcpy(past_value.data(), present_value.data(), present_value.get_byte_size());
            decoder_request_.set_tensor(entry.value_input_name, past_value);
        }
    }
}

void WhisperCppSUT::load() {
    if (loaded_) {
        return;
    }

    // Read encoder model
    encoder_model_ = core_.read_model(encoder_path_);
    map_encoder_names();

    // Read decoder model
    decoder_model_ = core_.read_model(decoder_path_);
    map_decoder_names();

    // Build compile properties
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;

    if (device_ == "CPU") {
        properties[ov::hint::enable_cpu_pinning.name()] = true;
    }

    // Compile encoder
    compiled_encoder_ = core_.compile_model(encoder_model_, device_, properties);

    // Compile decoder
    compiled_decoder_ = core_.compile_model(decoder_model_, device_, properties);

    // Get optimal number of inference requests
    try {
        optimal_nireq_ = compiled_encoder_.get_property(ov::optimal_number_of_infer_requests);
    } catch (...) {
        optimal_nireq_ = 1;
    }

    // Create inference requests
    encoder_request_ = compiled_encoder_.create_infer_request();
    decoder_request_ = compiled_decoder_.create_infer_request();

    loaded_ = true;
}

ov::Tensor WhisperCppSUT::run_encoder(const float* mel_features, size_t mel_size) {
    // Create input tensor
    ov::Shape input_shape = {1, static_cast<size_t>(n_mels_), static_cast<size_t>(n_frames_)};
    ov::Tensor mel_tensor(ov::element::f32, input_shape);

    // Copy mel features
    size_t copy_size = std::min(mel_size * sizeof(float), mel_tensor.get_byte_size());
    std::memcpy(mel_tensor.data<float>(), mel_features, copy_size);

    // Set input and run
    encoder_request_.set_tensor(encoder_input_name_, mel_tensor);
    encoder_request_.infer();

    // Get encoder output (clone to keep data)
    ov::Tensor output = encoder_request_.get_tensor(encoder_output_name_);

    // Create a copy of the tensor data
    ov::Tensor result(output.get_element_type(), output.get_shape());
    std::memcpy(result.data(), output.data(), output.get_byte_size());

    return result;
}

ov::Tensor WhisperCppSUT::run_decoder_step(const ov::Tensor& encoder_hidden,
                                            const std::vector<int64_t>& input_ids,
                                            bool is_first_step) {
    // Create decoder input_ids tensor
    ov::Shape ids_shape = {1, input_ids.size()};
    ov::Tensor ids_tensor(ov::element::i64, ids_shape);
    std::memcpy(ids_tensor.data<int64_t>(), input_ids.data(), input_ids.size() * sizeof(int64_t));

    // Set inputs
    decoder_request_.set_tensor(decoder_input_ids_name_, ids_tensor);
    decoder_request_.set_tensor(decoder_encoder_hidden_name_, encoder_hidden);

    // Set beam_idx for KV-cache reordering (greedy = single beam at index 0)
    if (has_beam_idx_) {
        ov::Shape beam_shape = {1};
        ov::Tensor beam_tensor(ov::element::i32, beam_shape);
        beam_tensor.data<int32_t>()[0] = 0;  // Greedy: always use beam 0
        decoder_request_.set_tensor(decoder_beam_idx_name_, beam_tensor);
    }

    // Initialize or update KV-cache
    if (has_kv_cache_) {
        if (is_first_step) {
            init_kv_cache();
        }
        // Note: KV-cache is updated after inference in greedy_decode
    }

    // Run decoder
    decoder_request_.infer();

    // Get logits output
    return decoder_request_.get_tensor(decoder_output_name_);
}

std::vector<int64_t> WhisperCppSUT::greedy_decode(const ov::Tensor& encoder_hidden) {
    // Initialize decoder input with special tokens
    std::vector<int64_t> decoder_input = {
        SOT_TOKEN,
        EN_TOKEN,
        TRANSCRIBE_TOKEN,
        NO_TIMESTAMPS_TOKEN
    };

    std::vector<int64_t> generated_tokens;

    // Debug: track decoding progress
    auto start_time = std::chrono::steady_clock::now();
    const int timeout_seconds = 30;  // Timeout per sample

    for (int step = 0; step < max_new_tokens_; ++step) {
        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > timeout_seconds) {
            std::cerr << "\n[Whisper] TIMEOUT at step " << step << ", forcing EOT" << std::endl;
            break;
        }

        bool is_first_step = (step == 0);

        // For KV-cache, only pass the last token after the first step
        std::vector<int64_t> step_input;
        if (has_kv_cache_ && !is_first_step) {
            // Only the newly generated token
            step_input = {decoder_input.back()};
        } else {
            // Full sequence for first step or non-KV-cache models
            step_input = decoder_input;
        }

        // Run decoder step
        ov::Tensor logits = run_decoder_step(encoder_hidden, step_input, is_first_step);

        // Update KV-cache for next step
        if (has_kv_cache_) {
            update_kv_cache();
        }

        // Get shape info
        auto shape = logits.get_shape();  // [batch, seq_len, vocab_size]
        size_t vocab_size = shape.back();
        size_t seq_len = shape.size() > 2 ? shape[1] : 1;

        // Get logits for the last position
        const float* logits_data = logits.data<float>();
        const float* last_logits = logits_data + (seq_len - 1) * vocab_size;

        // Greedy: find argmax
        int64_t next_token = 0;
        float max_val = last_logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (last_logits[i] > max_val) {
                max_val = last_logits[i];
                next_token = static_cast<int64_t>(i);
            }
        }

        // Check for EOT
        if (next_token == EOT_TOKEN) {
            break;
        }

        // Append token
        generated_tokens.push_back(next_token);
        decoder_input.push_back(next_token);
    }

    return generated_tokens;
}

std::vector<int64_t> WhisperCppSUT::process_sample(const float* mel_features,
                                                    size_t mel_size,
                                                    uint64_t query_id,
                                                    int sample_idx) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    std::lock_guard<std::mutex> lock(inference_mutex_);

    issued_count_++;

    // Run encoder
    ov::Tensor encoder_hidden = run_encoder(mel_features, mel_size);

    // Greedy decode
    std::vector<int64_t> tokens = greedy_decode(encoder_hidden);

    // Store prediction if needed
    if (store_predictions_) {
        WhisperPrediction pred;
        pred.tokens = tokens;

        {
            std::lock_guard<std::mutex> pred_lock(predictions_mutex_);
            predictions_[sample_idx] = std::move(pred);
        }
    }

    // Call response callback
    {
        std::lock_guard<std::mutex> cb_lock(callback_mutex_);
        if (response_callback_) {
            response_callback_(query_id, tokens.data(), tokens.size());
        }
    }

    completed_count_++;

    return tokens;
}

void WhisperCppSUT::start_async(const float* mel_features,
                                size_t mel_size,
                                uint64_t query_id,
                                int sample_idx) {
    // For now, just call sync version
    process_sample(mel_features, mel_size, query_id, sample_idx);
}

void WhisperCppSUT::wait_all() {
    while (pending_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void WhisperCppSUT::reset_counters() {
    wait_all();

    issued_count_ = 0;
    completed_count_ = 0;
    {
        std::lock_guard<std::mutex> lock(predictions_mutex_);
        predictions_.clear();
    }
}

std::unordered_map<int, WhisperPrediction> WhisperCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void WhisperCppSUT::set_response_callback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    response_callback_ = callback;
}

} // namespace mlperf_ov
