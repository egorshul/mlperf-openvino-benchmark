/**
 * C++ SUT for SSD-ResNet34 Object Detection on multi-die accelerators.
 *
 * SSD-ResNet34 multi-die for 1200x1200 input.
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
#include <test_settings.h>

namespace mlperf_ov {

class SSDResNet34MultiDieCppSUT;

// Per-die context
struct SSDResNet34DieContext {
    std::string device_name;
    ov::CompiledModel compiled_model;
    int optimal_nireq = 1;
    size_t request_start_idx = 0;
    size_t request_count = 0;
    std::atomic<size_t> pool_search_hint{0};
};

// SSD-ResNet34 prediction result
struct SSDResNet34MultiDiePrediction {
    std::vector<float> boxes;    // [N, 4] flattened
    std::vector<float> scores;   // [N]
    std::vector<float> labels;   // [N]
    int num_detections = 0;
};

// Inference request context
struct SSDResNet34MultiDieInferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;
    std::string die_name;
    size_t pool_id = 0;
    SSDResNet34MultiDieCppSUT* sut = nullptr;

    static constexpr int MAX_BATCH = 64;
    uint64_t query_ids[MAX_BATCH];
    int sample_indices[MAX_BATCH];
    int actual_batch_size = 0;
    int num_dummies = 0;
};

/**
 * Clean SUT implementation for SSD-ResNet34 on multi-die accelerators.
 */
class SSDResNet34MultiDieCppSUT {
public:
    // Note: SSD-ResNet34 has very large input (1200x1200x3 = 17.3MB vs ResNet 224x224x3 = 0.6MB)
    // so we use smaller defaults: nireq_multiplier=2 (vs 4 for ResNet)
    SSDResNet34MultiDieCppSUT(const std::string& model_path,
                               const std::string& device_prefix,
                               int batch_size = 1,
                               const std::unordered_map<std::string, std::string>& compile_properties = {},
                               bool use_nhwc_input = true,  // NHWC is default
                               int nireq_multiplier = 2);   // Lower for large model (vs 4 for ResNet)

    ~SSDResNet34MultiDieCppSUT();

    void load();
    void warmup(int iterations = 2);  // Warmup inference on all dies
    bool is_loaded() const { return loaded_; }
    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }
    std::vector<std::string> get_active_devices() const;
    int get_batch_size() const { return batch_size_; }
    int get_total_requests() const;
    std::string get_input_name() const { return input_name_; }
    std::string get_boxes_name() const { return boxes_name_; }
    std::string get_scores_name() const { return scores_name_; }
    std::string get_labels_name() const { return labels_name_; }

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
    std::unordered_map<int, SSDResNet34MultiDiePrediction> get_predictions() const;
    void clear_predictions();

    // Callback for Offline mode
    using BatchResponseCallback = std::function<void(const std::vector<uint64_t>& query_ids)>;
    void set_batch_response_callback(BatchResponseCallback callback);

    // Direct LoadGen mode (Server)
    void enable_direct_loadgen(bool enable);

    // Server mode: register sample data pointers
    void register_sample_data(int sample_idx, const float* data, size_t size);
    void clear_sample_data();

    // Server mode: fast dispatch (used by LoadGen SUT)
    void issue_queries_server_fast(const std::vector<uint64_t>& query_ids,
                                   const std::vector<int>& sample_indices);

    // Explicit batching configuration (Intel-style)
    void enable_explicit_batching(bool enable, int batch_size = 4, int timeout_us = 500);
    bool is_explicit_batching_enabled() const { return use_explicit_batching_; }

    // Set specific target devices (call before load())
    void set_target_devices(const std::vector<std::string>& devices) { target_devices_ = devices; }

    // Run pure C++ Server benchmark
    void run_server_benchmark(
        size_t total_sample_count,
        size_t performance_sample_count,
        const std::string& mlperf_conf_path,
        const std::string& user_conf_path,
        const std::string& log_output_dir,
        double target_qps = 0,
        int64_t target_latency_ns = 0,
        int64_t min_duration_ms = 0,
        int64_t min_query_count = 0,
        bool is_accuracy_mode = false);

private:
    // Config
    std::string model_path_;
    std::string device_prefix_;
    int batch_size_;
    std::unordered_map<std::string, std::string> compile_properties_;
    bool use_nhwc_input_;
    int nireq_multiplier_;
    std::vector<std::string> target_devices_;  // If set, use these instead of discovering

    // OpenVINO
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    std::vector<std::unique_ptr<SSDResNet34DieContext>> die_contexts_;
    std::vector<std::string> active_devices_;

    // Request pool - smaller than ResNet due to larger input size (17.3MB vs 0.6MB)
    // 256 requests x 17.3MB = ~4.4GB memory for input buffers
    static constexpr int MAX_REQUESTS = 256;
    std::vector<std::unique_ptr<SSDResNet34MultiDieInferContext>> infer_contexts_;
    std::atomic<int> request_slots_[MAX_REQUESTS];
    std::atomic<size_t> pool_search_hint_{0};

    // Work queue - smaller than ResNet (lower throughput due to larger model)
    static constexpr int WORK_QUEUE_SIZE = 2048;
    struct WorkItem {
        uint64_t query_id;
        int sample_idx;
        std::atomic<bool> valid{false};
    };
    WorkItem work_queue_[WORK_QUEUE_SIZE];
    std::atomic<size_t> work_head_{0};
    std::atomic<size_t> work_tail_{0};

    // Issue threads (one per die)
    std::vector<std::thread> issue_threads_;
    std::atomic<bool> issue_running_{false};
    void issue_thread_func(size_t die_idx);

    // Explicit batching (Intel-style) - smaller batch for large model
    bool use_explicit_batching_ = false;
    int explicit_batch_size_ = 2;     // Smaller than ResNet (4) due to 17.3MB input
    int batch_timeout_us_ = 1000;     // Longer timeout (1ms) for larger data copy
    std::thread batcher_thread_;
    std::atomic<bool> batcher_running_{false};

    // Per-die batch queues - smaller than ResNet due to lower throughput
    static constexpr int BATCH_QUEUE_SIZE = 128;
    static constexpr int MAX_DIES = 16;
    struct BatchItem {
        uint64_t query_ids[64];
        int sample_indices[64];
        int actual_size = 0;
        int num_dummies = 0;
        std::atomic<bool> valid{false};
    };
    // Separate queue for each die - no contention!
    BatchItem batch_queues_[MAX_DIES][BATCH_QUEUE_SIZE];
    std::atomic<size_t> batch_heads_[MAX_DIES];
    std::atomic<size_t> batch_tails_[MAX_DIES];
    std::atomic<size_t> next_die_{0};  // Round-robin counter

    void batcher_thread_func();
    void issue_thread_batched_func(size_t die_idx);

    // Model info
    std::string input_name_;
    std::string boxes_name_;
    std::string scores_name_;
    std::string labels_name_;
    int boxes_idx_ = 0;
    int scores_idx_ = 1;
    int labels_idx_ = 2;
    ov::Shape input_shape_;
    ov::element::Type input_type_;
    size_t input_byte_size_ = 0;

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<uint64_t> queued_count_{0};

    // Predictions
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, SSDResNet34MultiDiePrediction> predictions_;

    // Response handling
    BatchResponseCallback batch_response_callback_;
    std::mutex callback_mutex_;
    std::atomic<bool> use_direct_loadgen_{false};

    // Sample data cache
    struct SampleData {
        const float* data;
        size_t size;
    };
    mutable std::shared_mutex sample_cache_mutex_;
    std::unordered_map<int, SampleData> sample_data_cache_;

    // Helpers
    std::vector<std::string> discover_dies();
    ov::AnyMap build_compile_properties();
    size_t acquire_request();
    void release_request(size_t id);
    size_t acquire_request_for_die(size_t die_idx);
    void on_inference_complete(SSDResNet34MultiDieInferContext* ctx);
    void map_output_names();

    friend class SSDResNet34ServerSUT;
};

// =============================================================================
// PURE C++ SERVER SUT
// =============================================================================

/**
 * Pure C++ SUT for Server mode - registered directly with LoadGen.
 * Uses work queue for non-blocking dispatch.
 */
class SSDResNet34ServerSUT : public mlperf::SystemUnderTest {
public:
    explicit SSDResNet34ServerSUT(SSDResNet34MultiDieCppSUT* backend, const std::string& name = "SSDResNet34ServerSUT")
        : backend_(backend), name_(name) {
        backend_->enable_direct_loadgen(true);
    }

    ~SSDResNet34ServerSUT() override = default;

    const std::string& Name() override { return name_; }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        // Fast dispatch via work queue
        for (const auto& sample : samples) {
            enqueue_work(sample.id, sample.index);
        }
    }

    void FlushQueries() override {
        backend_->wait_all();
    }

private:
    void enqueue_work(uint64_t query_id, int sample_idx) {
        size_t head = backend_->work_head_.fetch_add(1, std::memory_order_acq_rel);
        size_t idx = head % SSDResNet34MultiDieCppSUT::WORK_QUEUE_SIZE;

        // Brief spin if slot busy (rare)
        while (backend_->work_queue_[idx].valid.load(std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        }

        backend_->work_queue_[idx].query_id = query_id;
        backend_->work_queue_[idx].sample_idx = sample_idx;
        backend_->work_queue_[idx].valid.store(true, std::memory_order_release);
        backend_->queued_count_.fetch_add(1, std::memory_order_relaxed);
    }

    SSDResNet34MultiDieCppSUT* backend_;
    std::string name_;
};

/**
 * Pure C++ QSL - samples pre-registered via register_sample_data()
 */
class SSDResNet34ServerQSL : public mlperf::QuerySampleLibrary {
public:
    SSDResNet34ServerQSL(size_t total_count, size_t perf_count)
        : total_count_(total_count), perf_count_(perf_count), name_("SSDResNet34ServerQSL") {}

    const std::string& Name() override { return name_; }
    size_t TotalSampleCount() override { return total_count_; }
    size_t PerformanceSampleCount() override { return perf_count_; }
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>&) override {}
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>&) override {}

private:
    size_t total_count_;
    size_t perf_count_;
    std::string name_;
};

} // namespace mlperf_ov
