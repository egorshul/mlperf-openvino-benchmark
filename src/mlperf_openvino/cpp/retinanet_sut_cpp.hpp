/**
 * C++ RetinaNet SUT implementation for OpenVINO.
 *
 * Specialized for RetinaNet Object Detection with:
 * - 1 input: float32 image [1, 3, 800, 800]
 * - 3 outputs: boxes, scores, labels
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
class RetinaNetCppSUT;

/**
 * Context for each RetinaNet inference request.
 */
struct RetinaNetInferContext {
    ov::InferRequest request;

    // Input tensor (keep alive during inference)
    ov::Tensor input_tensor;

    uint64_t query_id = 0;
    int sample_idx = 0;
    size_t pool_id = 0;
    RetinaNetCppSUT* sut = nullptr;
};

/**
 * RetinaNet prediction result.
 */
struct RetinaNetPrediction {
    std::vector<float> boxes;    // [N, 4] flattened
    std::vector<float> scores;   // [N]
    std::vector<float> labels;   // [N]
    int num_detections = 0;
};

/**
 * High-performance RetinaNet SUT using C++ InferRequest pool.
 */
class RetinaNetCppSUT {
public:
    RetinaNetCppSUT(const std::string& model_path,
                    const std::string& device = "CPU",
                    int num_streams = 0,
                    const std::string& performance_hint = "THROUGHPUT",
                    const std::string& input_layout = "");

    ~RetinaNetCppSUT();

    void load();
    bool is_loaded() const { return loaded_; }
    int get_optimal_nireq() const { return optimal_nireq_; }

    // Input/output info
    std::string get_input_name() const { return input_name_; }
    std::string get_boxes_name() const { return boxes_name_; }
    std::string get_scores_name() const { return scores_name_; }
    std::string get_labels_name() const { return labels_name_; }

    // Get input shape
    std::vector<size_t> get_input_shape() const;

    /**
     * Start async inference.
     */
    void start_async(const float* input_data,
                     size_t input_size,
                     uint64_t query_id,
                     int sample_idx);

    void wait_all();

    uint64_t get_completed_count() const { return completed_count_.load(); }
    uint64_t get_issued_count() const { return issued_count_.load(); }

    void reset_counters();

    void set_store_predictions(bool store) { store_predictions_ = store; }
    std::unordered_map<int, RetinaNetPrediction> get_predictions() const;

    /**
     * Response callback receives: query_id, boxes_ptr, boxes_size,
     *                             scores_ptr, scores_size,
     *                             labels_ptr, labels_size
     */
    using ResponseCallback = std::function<void(
        uint64_t query_id,
        const float* boxes, size_t boxes_size,
        const float* scores, size_t scores_size,
        const float* labels, size_t labels_size)>;

    void set_response_callback(ResponseCallback callback);

    void on_inference_complete(RetinaNetInferContext* ctx);

private:
    std::string model_path_;
    std::string device_;
    int num_streams_;
    std::string performance_hint_;
    std::string input_layout_;

    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;

    std::vector<std::unique_ptr<RetinaNetInferContext>> infer_contexts_;

    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::queue<size_t> available_ids_;

    // Input info
    std::string input_name_;
    ov::Shape input_shape_;

    // Output info
    std::string boxes_name_;
    std::string scores_name_;
    std::string labels_name_;
    int boxes_idx_ = 0;
    int scores_idx_ = 1;
    int labels_idx_ = 2;

    int optimal_nireq_ = 1;

    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<int> callbacks_running_{0};

    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, RetinaNetPrediction> predictions_;

    ResponseCallback response_callback_;
    std::mutex callback_mutex_;

    size_t get_idle_request();
    void return_request(size_t id);
    void map_output_names();
};

} // namespace mlperf_ov
