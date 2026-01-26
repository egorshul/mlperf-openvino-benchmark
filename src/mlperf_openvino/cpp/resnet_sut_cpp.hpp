/**
 * C++ System Under Test (SUT) implementation for OpenVINO.
 *
 * This bypasses Python GIL limitations by handling all inference
 * callbacks in pure C++, similar to Intel/NVIDIA MLPerf submissions.
 *
 * Key optimizations:
 * - Multiple InferRequest pool with async execution
 * - Callbacks run in OpenVINO threads (no Python GIL)
 * - Lock-free counters where possible
 * - Minimal memory allocations in hot path
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
class ResNetCppSUT;

/**
 * Context for each inference request.
 */
struct InferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;  // Keep tensor alive during inference
    uint64_t query_id = 0;
    int sample_idx = 0;
    size_t pool_id = 0;
    ResNetCppSUT* sut = nullptr;
};

/**
 * High-performance SUT using C++ InferRequest pool.
 *
 * Bypasses Python GIL by handling inference callbacks entirely in C++.
 */
class ResNetCppSUT {
public:
    /**
     * Constructor.
     *
     * @param model_path Path to ONNX or OpenVINO IR model
     * @param device Target device (CPU, GPU, etc.)
     * @param num_streams Number of inference streams (0 = auto)
     * @param performance_hint Performance hint (THROUGHPUT or LATENCY)
     */
    ResNetCppSUT(const std::string& model_path,
           const std::string& device = "CPU",
           int num_streams = 0,
           const std::string& performance_hint = "THROUGHPUT",
           bool use_nhwc_input = true);  // NHWC is default

    ~ResNetCppSUT();

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
     * Get input tensor name.
     */
    std::string get_input_name() const;

    /**
     * Get output tensor name.
     */
    std::string get_output_name() const;

    /**
     * Start async inference with callback.
     *
     * This method is called from Python but executes callback in C++
     * without holding GIL.
     *
     * @param input_data Pointer to input data
     * @param input_size Size of input data in bytes
     * @param query_id LoadGen query ID
     * @param sample_idx Sample index
     */
    void start_async(const float* input_data,
                     size_t input_size,
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
     */
    std::unordered_map<int, std::vector<float>> get_predictions() const;

    /**
     * Set the callback function for LoadGen responses.
     * This is called from Python to register the QuerySamplesComplete callback.
     */
    using ResponseCallback = std::function<void(uint64_t query_id, const float* data, size_t size)>;
    void set_response_callback(ResponseCallback callback);

    /**
     * Called when inference completes - handles response.
     */
    void on_inference_complete(InferContext* ctx);

private:
    // Model configuration
    std::string model_path_;
    std::string device_;
    int num_streams_;
    std::string performance_hint_;
    bool use_nhwc_input_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;

    // InferRequest pool
    std::vector<std::unique_ptr<InferContext>> infer_contexts_;

    // Pool management
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::queue<size_t> available_ids_;

    // Model info
    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::element::Type input_type_;
    int optimal_nireq_;

    // State
    bool loaded_;
    std::atomic<uint64_t> issued_count_;
    std::atomic<uint64_t> completed_count_;
    std::atomic<int> pending_count_;
    std::atomic<int> callbacks_running_;  // Track callbacks still executing

    // Cached output info (set during load for thread-safe access)
    size_t output_idx_;  // Which output tensor to use (0 or 1)

    // Predictions storage (for accuracy mode)
    bool store_predictions_;
    mutable std::mutex predictions_mutex_;  // mutable for const get_predictions()
    std::unordered_map<int, std::vector<float>> predictions_;

    // Response callback (calls Python's QuerySamplesComplete)
    ResponseCallback response_callback_;
    std::mutex callback_mutex_;

    // Get available InferRequest from pool (blocks if none available)
    size_t get_idle_request();

    // Return InferRequest to pool
    void return_request(size_t id);
};

} // namespace mlperf_ov
