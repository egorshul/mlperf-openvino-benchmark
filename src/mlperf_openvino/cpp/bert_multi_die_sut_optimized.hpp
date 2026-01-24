/**
 * Optimized BERT SUT for multi-die accelerators with dynamic sequence length buckets.
 *
 * Key optimizations:
 * - Sequence length buckets: [128, 192, 256, 384] to minimize padding
 * - Optimal batch sizes per bucket: short sequences -> larger batches
 * - Multiple pre-compiled models for each (batch_size, seq_length) combination
 * - Pre-staged buffers: samples pre-organized by bucket for fast batch copy
 * - Per-bucket nireq: more inference requests for busier buckets
 * - Per-die request pools for load balancing
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
#include <condition_variable>
#include <queue>

#include <openvino/openvino.hpp>

#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

namespace mlperf_ov {

// =============================================================================
// CONFIGURATION
// =============================================================================

// Sequence length buckets based on distribution analysis
// Bucket 0: seq <= 128 (~21%), Bucket 1: seq <= 165 (~31%)
// Bucket 2: seq <= 256 (~37%), Bucket 3: seq <= 384 (~11%)
constexpr int SEQ_BUCKETS[] = {128, 165, 256, 384};
constexpr int NUM_SEQ_BUCKETS = 4;

// Optimal batch sizes per bucket (smaller batch for longer sequences to save memory)
constexpr int BATCH_SIZES[] = {4, 4, 2, 1};

// Per-bucket nireq multipliers (more requests for busier buckets 1 and 2)
constexpr int NIREQ_MULTIPLIERS[] = {1, 2, 2, 1};

// Model configuration key
struct BertModelConfig {
    int batch_size;
    int seq_length;

    bool operator==(const BertModelConfig& other) const {
        return batch_size == other.batch_size && seq_length == other.seq_length;
    }
};

// Hash function for BertModelConfig
struct BertModelConfigHash {
    size_t operator()(const BertModelConfig& config) const {
        return std::hash<int>()(config.batch_size) ^
               (std::hash<int>()(config.seq_length) << 16);
    }
};

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================

class BertOptimizedSUT;

// =============================================================================
// PER-MODEL CONTEXT
// =============================================================================

struct BertOptInferContext {
    ov::InferRequest request;
    ov::Tensor input_ids_tensor;
    ov::Tensor attention_mask_tensor;
    ov::Tensor token_type_ids_tensor;

    BertModelConfig config;
    size_t die_idx = 0;
    size_t pool_id = 0;
    BertOptimizedSUT* sut = nullptr;

    static constexpr int MAX_BATCH = 16;
    uint64_t query_ids[MAX_BATCH];
    int sample_indices[MAX_BATCH];
    int actual_batch_size = 0;
    int num_dummies = 0;
};

// Per-die, per-config compiled model and request pool
struct BertOptModelContext {
    BertModelConfig config;
    ov::CompiledModel compiled_model;
    std::vector<std::unique_ptr<BertOptInferContext>> requests;
    std::atomic<int>* slot_states = nullptr;
    size_t num_requests = 0;
    std::atomic<size_t> pool_hint{0};
};

// Per-die context
struct BertOptDieContext {
    std::string device_name;
    size_t die_idx;
    std::unordered_map<BertModelConfig, std::unique_ptr<BertOptModelContext>, BertModelConfigHash> models;
};

// =============================================================================
// PRE-STAGED BUFFER FOR FAST BATCH COPY (PINNED MEMORY)
// =============================================================================

struct BertStagedSample {
    int sample_idx;
    int actual_seq_len;
    size_t buffer_offset;  // Offset into bucket's staged buffer
};

// Pinned memory buffer for faster DMA transfers
struct PinnedBuffer {
    int64_t* data = nullptr;
    size_t size = 0;

    void allocate(size_t count);
    void free();
    ~PinnedBuffer() { free(); }

    // Non-copyable
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
};

struct BertStagedBucket {
    PinnedBuffer input_ids;           // Pinned buffer for all samples in bucket
    PinnedBuffer attention_mask;
    PinnedBuffer token_type_ids;
    std::vector<BertStagedSample> samples;  // Metadata for each sample
    std::unordered_map<int, size_t> sample_to_index;  // sample_idx -> index in samples
    int seq_length;  // Bucket sequence length
    bool staged = false;
};

// =============================================================================
// DOUBLE-BUFFERING FOR ASYNC PREFETCH
// =============================================================================

struct DoubleBatchBuffer {
    // Two buffers for ping-pong
    PinnedBuffer input_ids[2];
    PinnedBuffer attention_mask[2];
    PinnedBuffer token_type_ids[2];

    int current_buffer = 0;  // Which buffer is being used for inference
    int batch_size = 0;
    int seq_length = 0;
    bool allocated = false;

    void allocate(int batch, int seq_len);
    void swap() { current_buffer = 1 - current_buffer; }
    int next_buffer() const { return 1 - current_buffer; }
};

// =============================================================================
// WORK ITEM FOR BATCHING
// =============================================================================

struct BertOptWorkItem {
    uint64_t query_id;
    int sample_idx;
    int bucket_idx;
    int actual_seq_len;
};

struct BertOptBatch {
    std::vector<uint64_t> query_ids;
    std::vector<int> sample_indices;
    int bucket_idx;
    int target_batch_size;
    int num_dummies;
};

// =============================================================================
// SAMPLE DATA CACHE
// =============================================================================

struct BertOptSampleInfo {
    const int64_t* input_ids;
    const int64_t* attention_mask;
    const int64_t* token_type_ids;
    int actual_seq_len;
    int bucket_idx;
};

// =============================================================================
// PREDICTION RESULT
// =============================================================================

struct BertOptPrediction {
    std::vector<float> start_logits;
    std::vector<float> end_logits;
};

// =============================================================================
// OPTIMIZED BERT SUT
// =============================================================================

class BertOptimizedSUT {
public:
    BertOptimizedSUT(
        const std::string& model_path,
        const std::string& device_prefix,
        const std::unordered_map<std::string, std::string>& compile_properties = {},
        int nireq_per_config = 4);

    ~BertOptimizedSUT();

    // Configuration (call before load())
    void set_target_devices(const std::vector<std::string>& devices);
    void set_bucket_batch_sizes(const std::vector<int>& batch_sizes);
    void set_bucket_nireq_multipliers(const std::vector<int>& multipliers);

    // Load and compile all model variants
    void load();
    bool is_loaded() const { return loaded_; }

    // Info
    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }
    std::vector<std::string> get_active_devices() const;
    int get_num_model_configs() const;
    std::vector<std::pair<int, int>> get_model_configs() const;

    // Sample registration with sequence length info
    void register_sample(int sample_idx,
                         const int64_t* input_ids,
                         const int64_t* attention_mask,
                         const int64_t* token_type_ids,
                         int actual_seq_len);
    void clear_samples();

    // Stage samples into contiguous buffers per bucket (call after all samples registered)
    void stage_samples();

    // Compute bucket for a sequence length
    static int get_bucket_index(int seq_len);
    static int get_bucket_seq_len(int bucket_idx);

    // Offline mode: submit batch with pre-grouped samples
    void submit_batch(int bucket_idx,
                      const std::vector<uint64_t>& query_ids,
                      const std::vector<int>& sample_indices);

    // Server mode: fast dispatch (uses internal batching)
    void issue_queries(const std::vector<uint64_t>& query_ids,
                       const std::vector<int>& sample_indices);

    // Wait and stats
    void wait_all();
    void reset_counters();
    uint64_t get_completed_count() const { return completed_count_.load(); }
    uint64_t get_issued_count() const { return issued_count_.load(); }

    // Predictions
    void set_store_predictions(bool store) { store_predictions_ = store; }
    std::unordered_map<int, BertOptPrediction> get_predictions() const;
    void clear_predictions();

    // Direct LoadGen mode
    void enable_direct_loadgen(bool enable) { use_direct_loadgen_.store(enable); }

    // Server batching config
    void set_batching_timeout_us(int timeout_us) { batch_timeout_us_ = timeout_us; }
    void set_min_batch_size(int min_batch) { min_batch_size_ = min_batch; }

private:
    // Config
    std::string model_path_;
    std::string device_prefix_;
    std::unordered_map<std::string, std::string> compile_properties_;
    int nireq_per_config_;
    std::vector<std::string> target_devices_;
    std::vector<int> bucket_batch_sizes_;
    std::vector<int> bucket_nireq_multipliers_;

    // OpenVINO
    ov::Core core_;
    std::shared_ptr<ov::Model> base_model_;
    std::vector<std::unique_ptr<BertOptDieContext>> die_contexts_;

    // Input/output names
    std::string input_ids_name_;
    std::string attention_mask_name_;
    std::string token_type_ids_name_;
    std::string start_logits_name_;
    std::string end_logits_name_;
    bool single_output_ = false;
    size_t start_output_idx_ = 0;
    size_t end_output_idx_ = 1;

    // Request slot management
    static constexpr int SLOT_FREE = -1;
    std::unique_ptr<std::atomic<int>[]> all_slot_states_;
    size_t total_slots_ = 0;

    // Sample cache (used before staging)
    mutable std::shared_mutex sample_mutex_;
    std::unordered_map<int, BertOptSampleInfo> samples_;

    // Pre-staged buffers per bucket (used after stage_samples() called)
    BertStagedBucket staged_buckets_[NUM_SEQ_BUCKETS];
    bool samples_staged_ = false;

    // Double-buffering for async prefetch (one per bucket per die)
    std::vector<std::vector<DoubleBatchBuffer>> double_buffers_;  // [die_idx][bucket_idx]

    // Batching for Server mode
    int batch_timeout_us_ = 500;
    int min_batch_size_ = 1;

    // Per-bucket work queues for Server mode
    static constexpr int QUEUE_SIZE = 1024;
    struct BucketQueue {
        BertOptWorkItem items[QUEUE_SIZE];
        std::atomic<size_t> head{0};
        std::atomic<size_t> tail{0};
    };
    BucketQueue bucket_queues_[NUM_SEQ_BUCKETS];

    // Batcher thread for Server mode
    std::thread batcher_thread_;
    std::atomic<bool> batcher_running_{false};
    void batcher_thread_func();

    // Per-bucket dispatch threads
    std::vector<std::thread> dispatch_threads_;
    std::atomic<bool> dispatch_running_{false};

    // Per-bucket batch queues
    static constexpr int BATCH_QUEUE_SIZE = 128;
    struct BatchQueue {
        BertOptBatch batches[BATCH_QUEUE_SIZE];
        std::atomic<size_t> head{0};
        std::atomic<size_t> tail{0};
        std::atomic<bool> valid[BATCH_QUEUE_SIZE];
    };
    BatchQueue batch_queues_[NUM_SEQ_BUCKETS];

    void dispatch_thread_func(int bucket_idx);

    // Die round-robin for each bucket
    std::atomic<size_t> die_round_robin_[NUM_SEQ_BUCKETS];

    // State
    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};

    // Predictions
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, BertOptPrediction> predictions_;

    // LoadGen mode
    std::atomic<bool> use_direct_loadgen_{false};

    // Helpers
    std::vector<std::string> discover_devices();
    ov::AnyMap build_compile_properties();
    void map_input_output_names();
    std::shared_ptr<ov::Model> reshape_model(int batch_size, int seq_length);

    size_t acquire_request(size_t die_idx, const BertModelConfig& config);
    void release_request(size_t die_idx, const BertModelConfig& config, size_t pool_id);

    void on_inference_complete(BertOptInferContext* ctx);

    // Optimized copy using staged buffers
    void copy_sample_to_tensor(int sample_idx, int bucket_seq_len,
                               int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                               int offset_in_batch);
    void copy_staged_sample_to_tensor(int bucket_idx, size_t staged_idx, int bucket_seq_len,
                                      int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                                      int offset_in_batch);

    friend class BertOptimizedServerSUT;
};

// =============================================================================
// LOADGEN SUT WRAPPER
// =============================================================================

class BertOptimizedServerSUT : public mlperf::SystemUnderTest {
public:
    explicit BertOptimizedServerSUT(BertOptimizedSUT* backend, const std::string& name = "BertOptimizedSUT")
        : backend_(backend), name_(name) {
        backend_->enable_direct_loadgen(true);
    }

    const std::string& Name() override { return name_; }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        std::vector<uint64_t> query_ids;
        std::vector<int> sample_indices;
        query_ids.reserve(samples.size());
        sample_indices.reserve(samples.size());

        for (const auto& s : samples) {
            query_ids.push_back(s.id);
            sample_indices.push_back(s.index);
        }
        backend_->issue_queries(query_ids, sample_indices);
    }

    void FlushQueries() override {
        backend_->wait_all();
    }

private:
    BertOptimizedSUT* backend_;
    std::string name_;
};

// =============================================================================
// LOADGEN QSL WRAPPER
// =============================================================================

class BertOptimizedQSL : public mlperf::QuerySampleLibrary {
public:
    BertOptimizedQSL(size_t total, size_t perf)
        : total_(total), perf_(perf), name_("BertOptimizedQSL") {}

    const std::string& Name() override { return name_; }
    size_t TotalSampleCount() override { return total_; }
    size_t PerformanceSampleCount() override { return perf_; }
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>&) override {}
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>&) override {}

private:
    size_t total_, perf_;
    std::string name_;
};

} // namespace mlperf_ov
