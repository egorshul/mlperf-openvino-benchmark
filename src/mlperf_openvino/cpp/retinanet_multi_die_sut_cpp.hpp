/**
 * C++ RetinaNet SUT implementation for multi-die accelerators.
 *
 * Key design principles:
 * 1. Compile model once for each die
 * 2. Round-robin distribution for load balancing
 * 3. Handle 3 outputs: boxes, scores, labels
 * 4. NHWC input support via PrePostProcessor
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace mlperf_ov {

// Forward declaration
class RetinaNetMultiDieCppSUT;

/**
 * RetinaNet prediction result.
 */
struct RetinaNetMultiDiePrediction {
    std::vector<float> boxes;    // [N, 4] flattened
    std::vector<float> scores;   // [N]
    std::vector<float> labels;   // [N]
    int num_detections = 0;
};

/**
 * Context for a single die.
 */
struct RetinaNetDieContext {
    std::string device_name;
    ov::CompiledModel compiled_model;
    int optimal_nireq = 1;
};

/**
 * Context for each inference request.
 */
struct RetinaNetMultiDieInferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;

    uint64_t query_id = 0;
    int sample_idx = 0;
    size_t pool_id = 0;
    std::string die_name;
    RetinaNetMultiDieCppSUT* sut = nullptr;
};

/**
 * High-performance RetinaNet SUT for multi-die accelerators.
 */
class RetinaNetMultiDieCppSUT {
public:
    /**
     * Constructor.
     *
     * @param model_path Path to ONNX or OpenVINO IR model
     * @param device_prefix Device prefix (e.g., "NPU", "VPU")
     * @param compile_properties Device-specific compile properties
     * @param use_nhwc_input If true, expect NHWC input and add transpose to model
     */
    RetinaNetMultiDieCppSUT(const std::string& model_path,
                            const std::string& device_prefix,
                            const std::unordered_map<std::string, std::string>& compile_properties = {},
                            bool use_nhwc_input = false);

    ~RetinaNetMultiDieCppSUT();

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
     * Get total number of inference requests.
     */
    int get_total_requests() const { return static_cast<int>(infer_contexts_.size()); }

    /**
     * Get input name.
     */
    std::string get_input_name() const { return input_name_; }

    /**
     * Get input shape.
     */
    std::vector<size_t> get_input_shape() const {
        return std::vector<size_t>(input_shape_.begin(), input_shape_.end());
    }

    /**
     * Start async inference.
     */
    void start_async(const float* input_data,
                     size_t input_size,
                     uint64_t query_id,
                     int sample_idx);

    /**
     * Wait for all pending inferences.
     */
    void wait_all();

    /**
     * Get completed count.
     */
    uint64_t get_completed_count() const { return completed_count_.load(); }

    /**
     * Get issued count.
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
     * Get stored predictions.
     */
    std::unordered_map<int, RetinaNetMultiDiePrediction> get_predictions() const;

    /**
     * Clear stored predictions.
     */
    void clear_predictions();

    /**
     * Response callback type.
     */
    using ResponseCallback = std::function<void(
        uint64_t query_id,
        const float* boxes, size_t boxes_size,
        const float* scores, size_t scores_size,
        const float* labels, size_t labels_size)>;

    /**
     * Set the callback function for LoadGen responses.
     */
    void set_response_callback(ResponseCallback callback);

    /**
     * Called when inference completes.
     */
    void on_inference_complete(RetinaNetMultiDieInferContext* ctx);

private:
    // Model configuration
    std::string model_path_;
    std::string device_prefix_;
    std::unordered_map<std::string, std::string> compile_properties_;
    bool use_nhwc_input_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;

    // Per-die contexts
    std::vector<std::unique_ptr<RetinaNetDieContext>> die_contexts_;
    std::vector<std::string> active_devices_;

    // InferRequest pool
    std::vector<std::unique_ptr<RetinaNetMultiDieInferContext>> infer_contexts_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::queue<size_t> available_ids_;

    // Model info
    std::string input_name_;
    ov::Shape input_shape_;
    ov::element::Type input_type_;

    // Output info
    std::string boxes_name_;
    std::string scores_name_;
    std::string labels_name_;
    int boxes_idx_ = 0;
    int scores_idx_ = 1;
    int labels_idx_ = 2;

    // State
    bool loaded_ = false;
    std::atomic<size_t> die_index_{0};
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<int> callbacks_running_{0};

    // Predictions storage
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, RetinaNetMultiDiePrediction> predictions_;

    // Response callback
    ResponseCallback response_callback_;
    std::mutex callback_mutex_;

    // Internal methods
    std::vector<std::string> discover_dies();
    size_t get_idle_request();
    void return_request(size_t id);
    ov::AnyMap build_compile_properties();
    void map_output_names();
};

} // namespace mlperf_ov
