/**
 * C++ SUT for ResNet50 on multi-die accelerators with maximum throughput.
 *
 * Key features:
 * - Multiple compiled models (one per die)
 * - Round-robin distribution across dies
 * - Proper batching support
 * - All callbacks in C++ without GIL
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
class ResNetMultiDieCppSUT;

/**
 * Context for a single die.
 */
struct DieContext {
    std::string device_name;
    ov::CompiledModel compiled_model;
    int optimal_nireq = 1;
};

/**
 * Context for each inference request.
 */
struct ResNetMultiDieInferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;
    std::string die_name;
    size_t pool_id = 0;
    ResNetMultiDieCppSUT* sut = nullptr;

    // Batch info
    std::vector<uint64_t> query_ids;
    std::vector<int> sample_indices;
    int actual_batch_size = 0;
};

/**
 * High-performance SUT for multi-die accelerators.
 *
 * Distributes inference across all available dies using round-robin.
 * Each die has its own compiled model and request pool.
 */
class ResNetMultiDieCppSUT {
public:
    /**
     * Constructor.
     *
     * @param model_path Path to ONNX or OpenVINO IR model
     * @param device_prefix Device prefix (e.g., "NPU", "VPU")
     * @param batch_size Batch size for inference
     * @param compile_properties Device-specific compile properties
     * @param use_nhwc_input If true, expect NHWC input and add transpose to model
     */
    ResNetMultiDieCppSUT(const std::string& model_path,
                   const std::string& device_prefix,
                   int batch_size = 1,
                   const std::unordered_map<std::string, std::string>& compile_properties = {},
                   bool use_nhwc_input = false);

    ~ResNetMultiDieCppSUT();

    /**
     * Load model and compile for all available dies.
     */
    void load();

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get number of active dies.
     */
    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }

    /**
     * Get list of active device names.
     */
    std::vector<std::string> get_active_devices() const;

    /**
     * Get batch size.
     */
    int get_batch_size() const { return batch_size_; }

    /**
     * Get total number of inference requests across all dies.
     */
    int get_total_requests() const;

    /**
     * Get input tensor name.
     */
    std::string get_input_name() const { return input_name_; }

    /**
     * Get output tensor name.
     */
    std::string get_output_name() const { return output_name_; }

    /**
     * Start async inference for a batch.
     *
     * @param input_data Pointer to batched input data [batch_size, C, H, W]
     * @param input_size Size of input data in bytes
     * @param query_ids Vector of query IDs for the batch
     * @param sample_indices Vector of sample indices for the batch
     * @param actual_batch_size Actual number of samples (may be < batch_size for last batch)
     */
    void start_async_batch(const float* input_data,
                           size_t input_size,
                           const std::vector<uint64_t>& query_ids,
                           const std::vector<int>& sample_indices,
                           int actual_batch_size);

    /**
     * Process multiple samples efficiently (for Server mode).
     * All samples are dispatched in C++ without returning to Python.
     *
     * @param all_input_data Vector of pointers to input data for each sample
     * @param input_sizes Vector of input sizes
     * @param query_ids Vector of query IDs
     * @param sample_indices Vector of sample indices
     */
    void issue_queries_server(const std::vector<const float*>& all_input_data,
                              const std::vector<size_t>& input_sizes,
                              const std::vector<uint64_t>& query_ids,
                              const std::vector<int>& sample_indices);

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
     * Reset counters and state.
     */
    void reset_counters();

    /**
     * Enable/disable storing predictions.
     */
    void set_store_predictions(bool store) { store_predictions_ = store; }

    /**
     * Get stored predictions (for accuracy mode).
     */
    std::unordered_map<int, std::vector<float>> get_predictions() const;

    /**
     * Clear stored predictions.
     */
    void clear_predictions();

    /**
     * Callback type for single sample response.
     * Called with (query_id, output_data, output_size) for each sample.
     */
    using ResponseCallback = std::function<void(uint64_t query_id, const float* data, size_t size)>;

    /**
     * Callback type for batch responses (more efficient for Server mode).
     * Called with vector of query_ids for the completed batch.
     */
    using BatchResponseCallback = std::function<void(const std::vector<uint64_t>& query_ids)>;

    /**
     * Set the callback function for LoadGen responses (per-sample).
     */
    void set_response_callback(ResponseCallback callback);

    /**
     * Set the batch callback function for LoadGen responses (more efficient).
     * When set, this is called instead of per-sample callback.
     */
    void set_batch_response_callback(BatchResponseCallback callback);

    /**
     * Called when inference completes - handles batch response.
     */
    void on_inference_complete(ResNetMultiDieInferContext* ctx);

private:
    // Model configuration
    std::string model_path_;
    std::string device_prefix_;
    int batch_size_;
    std::unordered_map<std::string, std::string> compile_properties_;
    bool use_nhwc_input_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;

    // Per-die contexts
    std::vector<std::unique_ptr<DieContext>> die_contexts_;
    std::vector<std::string> active_devices_;

    // InferRequest pool (shared across all dies)
    std::vector<std::unique_ptr<ResNetMultiDieInferContext>> infer_contexts_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::queue<size_t> available_ids_;

    // Round-robin die selection
    std::atomic<size_t> die_index_{0};

    // Model info
    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::element::Type input_type_;
    ov::element::Type output_type_;
    size_t output_idx_ = 0;
    size_t single_output_size_ = 0;  // Size of one sample's output

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<int> callbacks_running_{0};

    // Predictions storage
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, std::vector<float>> predictions_;

    // Response callbacks
    ResponseCallback response_callback_;
    BatchResponseCallback batch_response_callback_;
    std::mutex callback_mutex_;

    // Discover accelerator dies
    std::vector<std::string> discover_dies();

    // Build compile properties
    ov::AnyMap build_compile_properties();

    // Get idle request from pool
    size_t get_idle_request();

    // Return request to pool
    void return_request(size_t id);

    // Get next die (round-robin)
    const std::string& get_next_die();
};

} // namespace mlperf_ov
