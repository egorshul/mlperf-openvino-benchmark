/**
 * C++ BERT SUT implementation for OpenVINO.
 *
 * Specialized for BERT Question Answering with:
 * - 3 inputs: input_ids, attention_mask, token_type_ids (int64)
 * - 2 outputs: start_logits, end_logits (float32)
 *
 * Key optimizations:
 * - Multiple InferRequest pool with async execution
 * - Callbacks run in OpenVINO threads (no Python GIL)
 * - Lock-free counters where possible
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
class BertCppSUT;

/**
 * Context for each BERT inference request.
 */
struct BertInferContext {
    ov::InferRequest request;

    // Input tensors (keep alive during inference)
    ov::Tensor input_ids_tensor;
    ov::Tensor attention_mask_tensor;
    ov::Tensor token_type_ids_tensor;

    uint64_t query_id = 0;
    int sample_idx = 0;
    size_t pool_id = 0;
    BertCppSUT* sut = nullptr;
};

/**
 * BERT prediction result (start_logits, end_logits).
 */
struct BertPrediction {
    std::vector<float> start_logits;
    std::vector<float> end_logits;
};

/**
 * High-performance BERT SUT using C++ InferRequest pool.
 *
 * Bypasses Python GIL by handling inference callbacks entirely in C++.
 */
class BertCppSUT {
public:
    /**
     * Constructor.
     *
     * @param model_path Path to ONNX or OpenVINO IR model
     * @param device Target device (CPU, GPU, etc.)
     * @param num_streams Number of inference streams (0 = auto)
     * @param performance_hint Performance hint (THROUGHPUT or LATENCY)
     */
    BertCppSUT(const std::string& model_path,
               const std::string& device = "CPU",
               int num_streams = 0,
               const std::string& performance_hint = "THROUGHPUT");

    ~BertCppSUT();

    /**
     * Load and compile the model.
     */
    void load();

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get optimal number of inference requests.
     */
    int get_optimal_nireq() const { return optimal_nireq_; }

    /**
     * Get input tensor names.
     */
    std::string get_input_ids_name() const { return input_ids_name_; }
    std::string get_attention_mask_name() const { return attention_mask_name_; }
    std::string get_token_type_ids_name() const { return token_type_ids_name_; }

    /**
     * Get output tensor names.
     */
    std::string get_start_logits_name() const { return start_logits_name_; }
    std::string get_end_logits_name() const { return end_logits_name_; }

    /**
     * Get sequence length.
     */
    int get_seq_length() const { return seq_length_; }

    /**
     * Start async inference.
     *
     * @param input_ids Pointer to input_ids data (int64)
     * @param attention_mask Pointer to attention_mask data (int64)
     * @param token_type_ids Pointer to token_type_ids data (int64)
     * @param seq_length Sequence length
     * @param query_id LoadGen query ID
     * @param sample_idx Sample index
     */
    void start_async(const int64_t* input_ids,
                     const int64_t* attention_mask,
                     const int64_t* token_type_ids,
                     int seq_length,
                     uint64_t query_id,
                     int sample_idx);

    /**
     * Wait for all pending inferences to complete.
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
     * Reset counters.
     */
    void reset_counters();

    /**
     * Enable/disable storing predictions.
     */
    void set_store_predictions(bool store) { store_predictions_ = store; }

    /**
     * Get stored predictions (for accuracy mode).
     * Returns map of sample_idx -> (start_logits, end_logits)
     */
    std::unordered_map<int, BertPrediction> get_predictions() const;

    /**
     * Set the callback function for LoadGen responses.
     * Callback receives: query_id, start_logits_ptr, end_logits_ptr, logits_size
     */
    using ResponseCallback = std::function<void(uint64_t query_id,
                                                 const float* start_logits,
                                                 const float* end_logits,
                                                 size_t logits_size)>;
    void set_response_callback(ResponseCallback callback);

    /**
     * Called when inference completes - handles response.
     */
    void on_inference_complete(BertInferContext* ctx);

private:
    // Model configuration
    std::string model_path_;
    std::string device_;
    int num_streams_;
    std::string performance_hint_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;

    // InferRequest pool
    std::vector<std::unique_ptr<BertInferContext>> infer_contexts_;

    // Pool management
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::queue<size_t> available_ids_;

    // Model info - input names
    std::string input_ids_name_;
    std::string attention_mask_name_;
    std::string token_type_ids_name_;

    // Model info - output names
    std::string start_logits_name_;
    std::string end_logits_name_;
    bool single_output_ = false;  // True if model has combined output

    // Shape info
    int seq_length_ = 384;  // Default BERT sequence length
    int optimal_nireq_ = 1;

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<int> callbacks_running_{0};

    // Predictions storage (for accuracy mode)
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, BertPrediction> predictions_;

    // Response callback
    ResponseCallback response_callback_;
    std::mutex callback_mutex_;

    // Get available InferRequest from pool
    size_t get_idle_request();

    // Return InferRequest to pool
    void return_request(size_t id);

    // Map input/output names
    void map_input_names();
    void map_output_names();
};

} // namespace mlperf_ov
