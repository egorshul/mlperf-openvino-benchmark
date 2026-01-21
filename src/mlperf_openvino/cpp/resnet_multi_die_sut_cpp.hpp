/**
 * High-performance C++ SUT for ResNet50 on multi-die accelerators.
 *
 * Architecture (NVIDIA LWIS-style with per-die parallelism):
 * - IssueQuery: instant return, just pushes to shared work queue
 * - Per-die Issue Threads: each die has dedicated thread that pulls from
 *   shared queue, does memcpy, and submits to its own device (parallel!)
 * - Completion Thread: batches responses, calls QuerySamplesComplete
 * - Lock-free queues throughout (MPSC work queue, per-die request pools)
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

// LoadGen C++ API
#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>

namespace mlperf_ov {

class ResNetMultiDieCppSUT;

// Per-die context with its own request pool and issue thread
struct DieContext {
    std::string device_name;
    ov::CompiledModel compiled_model;
    int optimal_nireq = 1;

    // Per-die request pool indices (into main infer_contexts_ vector)
    size_t request_start_idx = 0;
    size_t request_count = 0;

    // Per-die lock-free pool management
    std::atomic<size_t> pool_search_hint{0};
};

// Inference request context with pre-allocated buffers
struct ResNetMultiDieInferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;
    std::string die_name;
    size_t pool_id = 0;
    ResNetMultiDieCppSUT* sut = nullptr;

    // Pre-allocated batch info (no dynamic allocation)
    static constexpr int MAX_BATCH = 64;
    uint64_t query_ids[MAX_BATCH];
    int sample_indices[MAX_BATCH];
    int actual_batch_size = 0;

    // Pre-allocated response buffer
    mlperf::QuerySampleResponse responses[MAX_BATCH];
};

/**
 * High-performance SUT with NVIDIA LWIS-style architecture.
 */
class ResNetMultiDieCppSUT {
public:
    ResNetMultiDieCppSUT(const std::string& model_path,
                         const std::string& device_prefix,
                         int batch_size = 1,
                         const std::unordered_map<std::string, std::string>& compile_properties = {},
                         bool use_nhwc_input = false);

    ~ResNetMultiDieCppSUT();

    void load();
    bool is_loaded() const { return loaded_; }
    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }
    std::vector<std::string> get_active_devices() const;
    int get_batch_size() const { return batch_size_; }
    int get_total_requests() const;
    std::string get_input_name() const { return input_name_; }
    std::string get_output_name() const { return output_name_; }

    // Offline mode batch dispatch
    void start_async_batch(const float* input_data,
                           size_t input_size,
                           const std::vector<uint64_t>& query_ids,
                           const std::vector<int>& sample_indices,
                           int actual_batch_size);

    void wait_all();
    uint64_t get_completed_count() const { return completed_count_.load(std::memory_order_relaxed); }
    uint64_t get_issued_count() const { return issued_count_.load(std::memory_order_relaxed); }
    void reset_counters();

    // Predictions for accuracy mode
    void set_store_predictions(bool store) { store_predictions_ = store; }
    std::unordered_map<int, std::vector<float>> get_predictions() const;
    void clear_predictions();

    // Callback for Offline mode
    using BatchResponseCallback = std::function<void(const std::vector<uint64_t>& query_ids)>;
    void set_batch_response_callback(BatchResponseCallback callback);

    // Direct LoadGen mode (Server)
    void enable_direct_loadgen(bool enable);

    // Inference complete handler
    void on_inference_complete(ResNetMultiDieInferContext* ctx);

    // Server mode: register sample data pointers
    void register_sample_data(int sample_idx, const float* data, size_t size);
    void clear_sample_data();

    // Server mode: INSTANT return - just pushes to work queue
    void issue_queries_server_fast(const std::vector<uint64_t>& query_ids,
                                   const std::vector<int>& sample_indices);

private:
    // Config
    std::string model_path_;
    std::string device_prefix_;
    int batch_size_;
    std::unordered_map<std::string, std::string> compile_properties_;
    bool use_nhwc_input_;

    // OpenVINO
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    std::vector<std::unique_ptr<DieContext>> die_contexts_;
    std::vector<std::string> active_devices_;

    // =========================================================================
    // LOCK-FREE REQUEST POOL
    // =========================================================================
    static constexpr int MAX_REQUESTS = 1024;
    std::vector<std::unique_ptr<ResNetMultiDieInferContext>> infer_contexts_;

    // Lock-free pool: each slot is either FREE (-1) or IN_USE (pool_id)
    std::atomic<int> request_slots_[MAX_REQUESTS];
    std::atomic<size_t> pool_search_hint_{0};

    // =========================================================================
    // WORK QUEUE (IssueQuery pushes here, Issue Thread pulls)
    // =========================================================================
    static constexpr int WORK_QUEUE_SIZE = 8192;

    struct WorkItem {
        uint64_t query_id;
        int sample_idx;
        std::atomic<bool> valid{false};
    };
    WorkItem work_queue_[WORK_QUEUE_SIZE];
    std::atomic<size_t> work_head_{0};  // IssueQuery writes here
    std::atomic<size_t> work_tail_{0};  // Issue thread reads here

    // Per-die issue threads (each pulls from shared queue, uses own die's requests)
    std::vector<std::thread> issue_threads_;
    std::atomic<bool> issue_running_{false};
    void issue_thread_func(size_t die_idx);

    // Per-die request acquisition (lock-free within die's pool)
    size_t acquire_request_for_die(size_t die_idx);
    void release_request_for_die(size_t die_idx, size_t local_id);

    // =========================================================================
    // COMPLETION THREAD WITH RESPONSE BATCHING
    // =========================================================================
    static constexpr int COMPLETION_QUEUE_SIZE = 4096;

    struct CompletionItem {
        ResNetMultiDieInferContext* ctx;
        std::atomic<bool> valid{false};
    };
    CompletionItem completion_queue_[COMPLETION_QUEUE_SIZE];
    std::atomic<size_t> completion_head_{0};
    std::atomic<size_t> completion_tail_{0};

    std::thread completion_thread_;
    std::atomic<bool> completion_running_{false};

    void completion_thread_func();
    void enqueue_completion(ResNetMultiDieInferContext* ctx);

    // =========================================================================
    // MODEL INFO
    // =========================================================================
    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::element::Type input_type_;
    ov::element::Type output_type_;
    size_t output_idx_ = 0;
    size_t single_output_size_ = 0;
    size_t input_byte_size_ = 0;

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<uint64_t> queued_count_{0};  // Samples in work queue

    // Predictions
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, std::vector<float>> predictions_;

    // Response handling
    BatchResponseCallback batch_response_callback_;
    std::mutex callback_mutex_;
    std::atomic<bool> use_direct_loadgen_{false};

    // Sample data cache (for fast Server mode) - THREAD SAFE
    struct SampleData {
        const float* data;
        size_t size;
    };
    mutable std::mutex sample_cache_mutex_;
    std::unordered_map<int, SampleData> sample_data_cache_;

    // Round-robin
    std::atomic<size_t> die_index_{0};

    // Helpers
    std::vector<std::string> discover_dies();
    ov::AnyMap build_compile_properties();

    // Lock-free request pool operations
    size_t acquire_request();
    void release_request(size_t id);
};

} // namespace mlperf_ov
