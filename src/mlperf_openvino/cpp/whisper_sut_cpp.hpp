/**
 * C++ Whisper SUT implementation for OpenVINO.
 *
 * Optimized for Whisper Large v3 encoder-decoder architecture:
 * - Encoder: mel spectrogram (float32) -> hidden states
 * - Decoder: autoregressive token generation with KV-cache
 *
 * Key optimizations:
 * - Encoder runs once per sample
 * - Decoder uses KV-cache for efficient autoregressive decoding
 * - Greedy decoding in C++ (no Python overhead)
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace mlperf_ov {

// Forward declaration
class WhisperCppSUT;

/**
 * Whisper prediction result (generated tokens + text).
 */
struct WhisperPrediction {
    std::vector<int64_t> tokens;
    std::string text;  // Decoded text (if tokenizer available)
};

/**
 * KV-cache entry: maps layer index to input/output names and shape info.
 */
struct KVCacheEntry {
    std::string key_input_name;
    std::string value_input_name;
    std::string key_output_name;
    std::string value_output_name;
    ov::Shape key_shape;
    ov::Shape value_shape;
};

/**
 * High-performance Whisper SUT using C++ for encoder-decoder inference.
 */
class WhisperCppSUT {
public:
    // Whisper special tokens
    static constexpr int64_t SOT_TOKEN = 50258;        // Start of transcript
    static constexpr int64_t EOT_TOKEN = 50257;        // End of transcript
    static constexpr int64_t TRANSCRIBE_TOKEN = 50359; // Transcribe task
    static constexpr int64_t NO_TIMESTAMPS_TOKEN = 50363; // No timestamps
    static constexpr int64_t EN_TOKEN = 50259;         // English language

    /**
     * Constructor.
     *
     * @param encoder_path Path to encoder OpenVINO IR model
     * @param decoder_path Path to decoder OpenVINO IR model
     * @param device Target device (CPU, GPU, etc.)
     * @param num_streams Number of inference streams (0 = auto)
     * @param max_new_tokens Maximum tokens to generate
     */
    WhisperCppSUT(const std::string& encoder_path,
                  const std::string& decoder_path,
                  const std::string& device = "CPU",
                  int num_streams = 0,
                  int max_new_tokens = 440);

    ~WhisperCppSUT();

    /**
     * Load and compile the models.
     */
    void load();

    /**
     * Check if models are loaded.
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get optimal number of inference requests.
     */
    int get_optimal_nireq() const { return optimal_nireq_; }

    /**
     * Get encoder input name.
     */
    std::string get_encoder_input_name() const { return encoder_input_name_; }

    /**
     * Get mel spectrogram dimensions.
     */
    int get_n_mels() const { return n_mels_; }
    int get_n_frames() const { return n_frames_; }

    /**
     * Check if decoder uses KV-cache.
     */
    bool has_kv_cache() const { return has_kv_cache_; }

    /**
     * Process a single audio sample (blocking).
     *
     * @param mel_features Pointer to mel spectrogram data (float32, shape [1, n_mels, n_frames])
     * @param mel_size Size of mel features in floats
     * @param query_id LoadGen query ID
     * @param sample_idx Sample index
     * @return Generated tokens
     */
    std::vector<int64_t> process_sample(const float* mel_features,
                                         size_t mel_size,
                                         uint64_t query_id,
                                         int sample_idx);

    /**
     * Start async inference (for future use).
     */
    void start_async(const float* mel_features,
                     size_t mel_size,
                     uint64_t query_id,
                     int sample_idx);

    /**
     * Wait for all pending inferences.
     */
    void wait_all();

    /**
     * Get number of completed samples.
     */
    uint64_t get_completed_count() const { return completed_count_.load(); }

    /**
     * Get number of issued samples.
     */
    uint64_t get_issued_count() const { return issued_count_.load(); }

    /**
     * Reset counters and predictions.
     */
    void reset_counters();

    /**
     * Enable/disable storing predictions.
     */
    void set_store_predictions(bool store) { store_predictions_ = store; }

    /**
     * Get stored predictions.
     */
    std::unordered_map<int, WhisperPrediction> get_predictions() const;

    /**
     * Set response callback.
     * Callback receives: query_id, tokens_ptr, tokens_size
     */
    using ResponseCallback = std::function<void(uint64_t query_id,
                                                 const int64_t* tokens,
                                                 size_t num_tokens)>;
    void set_response_callback(ResponseCallback callback);

private:
    // Model paths
    std::string encoder_path_;
    std::string decoder_path_;
    std::string device_;
    int num_streams_;
    int max_new_tokens_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> encoder_model_;
    std::shared_ptr<ov::Model> decoder_model_;
    ov::CompiledModel compiled_encoder_;
    ov::CompiledModel compiled_decoder_;

    // Inference requests
    ov::InferRequest encoder_request_;
    ov::InferRequest decoder_request_;

    // Model info - encoder
    std::string encoder_input_name_;
    std::string encoder_output_name_;
    int n_mels_ = 128;
    int n_frames_ = 3000;

    // Model info - decoder (basic inputs)
    std::string decoder_input_ids_name_;
    std::string decoder_encoder_hidden_name_;
    std::string decoder_output_name_;
    std::string decoder_beam_idx_name_;  // For KV-cache reordering

    // KV-cache support
    bool has_kv_cache_ = false;
    bool has_beam_idx_ = false;
    std::vector<KVCacheEntry> kv_cache_entries_;  // Per-layer KV cache info
    size_t num_layers_ = 0;
    size_t num_heads_ = 0;
    size_t head_dim_ = 0;

    // Shape info
    int optimal_nireq_ = 1;
    size_t encoder_hidden_size_ = 1280;  // Whisper Large
    size_t encoder_seq_len_ = 1500;      // 3000 frames / 2

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};

    // Predictions storage
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, WhisperPrediction> predictions_;

    // Response callback
    ResponseCallback response_callback_;
    std::mutex callback_mutex_;

    // Mutex for thread-safe inference
    std::mutex inference_mutex_;

    // Map input/output names
    void map_encoder_names();
    void map_decoder_names();

    // Run encoder
    ov::Tensor run_encoder(const float* mel_features, size_t mel_size);

    // Run decoder step (with KV-cache support)
    ov::Tensor run_decoder_step(const ov::Tensor& encoder_hidden,
                                 const std::vector<int64_t>& input_ids,
                                 bool is_first_step);

    // Initialize KV-cache tensors
    void init_kv_cache();

    // Copy present to past KV-cache
    void update_kv_cache();

    // Greedy decode
    std::vector<int64_t> greedy_decode(const ov::Tensor& encoder_hidden);
};

} // namespace mlperf_ov
