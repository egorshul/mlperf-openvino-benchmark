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
#include <cstring>
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

    // Run pure C++ Server benchmark (no Python in hot path!)
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
        bool enable_coalescing = false,
        int coalesce_batch_size = 8,
        int coalesce_window_us = 500);

    // DIRECT query processing for Server mode (minimum latency!)
    // Processes query immediately without queue - critical for latency SLA
    void process_query_direct(uint64_t query_id, int sample_idx);

    // BATCHED query processing for Server mode (higher throughput!)
    // Collects queries and processes them in batches
    void enqueue_for_coalescing(uint64_t query_id, int sample_idx);
    void process_coalesced_batch();
    void start_coalescing_thread(int batch_size, int window_us);
    void stop_coalescing_thread();

    // NON-BLOCKING enqueue for Server mode (used by ResNetServerSUT)
    void enqueue_work(uint64_t query_id, int sample_idx) {
        // Lock-free push to work queue - returns IMMEDIATELY
        size_t head = work_head_.fetch_add(1, std::memory_order_acq_rel);
        size_t idx = head % WORK_QUEUE_SIZE;

        // Spin if slot is still in use (very rare under normal load)
        while (work_queue_[idx].valid.load(std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            asm volatile("yield");
            #endif
        }

        work_queue_[idx].query_id = query_id;
        work_queue_[idx].sample_idx = sample_idx;
        work_queue_[idx].valid.store(true, std::memory_order_release);
        queued_count_.fetch_add(1, std::memory_order_relaxed);
    }

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
    mutable std::shared_mutex sample_cache_mutex_;  // Shared reads, exclusive writes
    std::unordered_map<int, SampleData> sample_data_cache_;

    // Round-robin
    std::atomic<size_t> die_index_{0};

    // =========================================================================
    // QUERY COALESCING (batches queries for higher throughput)
    // =========================================================================
    static constexpr int COALESCE_QUEUE_SIZE = 16384;

    struct CoalesceItem {
        uint64_t query_id;
        int sample_idx;
        std::chrono::steady_clock::time_point arrival_time;
        std::atomic<bool> valid{false};
    };
    CoalesceItem coalesce_queue_[COALESCE_QUEUE_SIZE];
    std::atomic<size_t> coalesce_head_{0};  // Writers push here
    std::atomic<size_t> coalesce_tail_{0};  // Coalescing thread reads here

    // Coalescing thread
    std::thread coalescing_thread_;
    std::atomic<bool> coalescing_running_{false};
    int coalesce_batch_size_ = 8;
    int coalesce_window_us_ = 500;  // 0.5ms default window

    void coalescing_thread_func();

    // Helpers
    std::vector<std::string> discover_dies();
    ov::AnyMap build_compile_properties();

    // Lock-free request pool operations
    size_t acquire_request();
    void release_request(size_t id);

    // Friend class for pure C++ SUT
    friend class ResNetServerSUT;
    friend class ResNetServerQSL;
};

} // namespace mlperf_ov

// =============================================================================
// PURE C++ SERVER SUT (eliminates Python from hot path)
// =============================================================================

namespace mlperf_ov {

/**
 * Pure C++ SUT for Server mode - registered directly with LoadGen.
 * Eliminates Python overhead in the hot path.
 * Uses direct processing (no batching) - lowest latency but lower throughput.
 */
class ResNetServerSUT : public mlperf::SystemUnderTest {
public:
    ResNetServerSUT(ResNetMultiDieCppSUT* backend, const std::string& name = "ResNetServerSUT")
        : backend_(backend), name_(name) {
        backend_->enable_direct_loadgen(true);
    }

    ~ResNetServerSUT() override = default;

    const std::string& Name() override { return name_; }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        // DIRECT PROCESSING for minimum latency (Server mode requires low latency!)
        // Queue adds latency → queries fail latency SLA → low QPS
        // Process each sample immediately instead.
        for (const auto& sample : samples) {
            backend_->process_query_direct(sample.id, sample.index);
        }
    }

    void FlushQueries() override {
        backend_->wait_all();
    }

private:
    ResNetMultiDieCppSUT* backend_;
    std::string name_;
};

/**
 * Coalescing SUT for Server mode - batches queries for higher throughput.
 * Collects queries over a short time window and processes them in batches.
 * Trades small amount of latency for significantly higher throughput.
 *
 * Key insight: NPU is ~3.5x more efficient with batch=8 vs batch=1
 * (6100 QPS vs 1742 QPS with 2 dies)
 */
class CoalescingServerSUT : public mlperf::SystemUnderTest {
public:
    CoalescingServerSUT(ResNetMultiDieCppSUT* backend,
                        int batch_size = 8,
                        int window_us = 500,
                        const std::string& name = "CoalescingServerSUT")
        : backend_(backend), name_(name), batch_size_(batch_size), window_us_(window_us) {
        backend_->enable_direct_loadgen(true);
    }

    ~CoalescingServerSUT() override {
        stop();
    }

    void start() {
        backend_->start_coalescing_thread(batch_size_, window_us_);
    }

    void stop() {
        backend_->stop_coalescing_thread();
    }

    const std::string& Name() override { return name_; }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        // Push to coalescing queue - thread will batch and process
        for (const auto& sample : samples) {
            backend_->enqueue_for_coalescing(sample.id, sample.index);
        }
    }

    void FlushQueries() override {
        // Process any remaining queries in the coalescing queue
        backend_->process_coalesced_batch();
        backend_->wait_all();
    }

private:
    ResNetMultiDieCppSUT* backend_;
    std::string name_;
    int batch_size_;
    int window_us_;
};

/**
 * Pure C++ QSL for Server mode - samples are pre-registered via register_sample_data()
 */
class ResNetServerQSL : public mlperf::QuerySampleLibrary {
public:
    ResNetServerQSL(size_t total_count, size_t perf_count)
        : total_count_(total_count), perf_count_(perf_count), name_("ResNetServerQSL") {}

    const std::string& Name() override { return name_; }
    size_t TotalSampleCount() override { return total_count_; }
    size_t PerformanceSampleCount() override { return perf_count_; }

    // Samples are already loaded by Python - these are no-ops
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {}
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {}

private:
    size_t total_count_;
    size_t perf_count_;
    std::string name_;
};

} // namespace mlperf_ov
