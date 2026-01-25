/**
 * BERT SUT for multi-die NPU accelerators.
 *
 * Sequence buckets: [128, 165, 256, 384]
 * Offline: batch [4,4,2,2] for throughput
 * Server: batch=1, direct inference for latency
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <loadgen.h>
#include <query_sample.h>

namespace mlperf_ov {

constexpr int SEQ_BUCKETS[] = {128, 165, 256, 384};
constexpr int NUM_SEQ_BUCKETS = 4;
constexpr int OFFLINE_BATCH_SIZES[] = {4, 4, 2, 2};
constexpr int SERVER_BATCH_SIZE = 1;

class BertOptimizedSUT;

struct BertOptInferContext {
    ov::InferRequest request;
    ov::Tensor input_ids_tensor;
    ov::Tensor attention_mask_tensor;
    ov::Tensor token_type_ids_tensor;

    int batch_size = 0;
    int seq_length = 0;
    int bucket_idx = 0;
    size_t die_idx = 0;
    size_t pool_id = 0;
    BertOptimizedSUT* sut = nullptr;

    static constexpr int MAX_BATCH = 16;
    uint64_t query_ids[MAX_BATCH];
    int sample_indices[MAX_BATCH];
    int actual_batch_size = 0;
    int num_dummies = 0;
};

struct BertBucketModelContext {
    ov::CompiledModel compiled_model;
    std::vector<std::unique_ptr<BertOptInferContext>> requests;
    std::atomic<int>* slot_states = nullptr;
    size_t num_requests = 0;
    std::atomic<size_t> pool_hint{0};
    int batch_size = 0;
    int seq_length = 0;
};

struct BertOptDieContext {
    std::string device_name;
    size_t die_idx = 0;
    std::unique_ptr<BertBucketModelContext> bucket_models[NUM_SEQ_BUCKETS];
};

struct BertSampleData {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    int actual_seq_len = 0;
    int bucket_idx = 0;
};

struct BertOptPrediction {
    std::vector<float> start_logits;
    std::vector<float> end_logits;
};

struct BertStagedSample {
    int sample_idx;
    int actual_seq_len;
    size_t buffer_offset;
};

struct BertStagedBucket {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    std::vector<BertStagedSample> samples;
    std::unordered_map<int, size_t> sample_to_index;
    int seq_length = 0;
    bool staged = false;
};

class BertOptimizedSUT {
public:
    BertOptimizedSUT(
        const std::string& model_path,
        const std::string& device_prefix,
        const std::unordered_map<std::string, std::string>& compile_properties = {},
        int nireq_per_bucket = 4);

    ~BertOptimizedSUT();

    void set_target_devices(const std::vector<std::string>& devices);
    void set_server_mode(bool server_mode) { server_mode_ = server_mode; }

    void load();
    void warmup(int iterations = 2);
    bool is_loaded() const { return loaded_; }

    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }
    std::vector<std::string> get_active_devices() const;
    int get_num_model_configs() const { return NUM_SEQ_BUCKETS; }
    std::vector<std::pair<int, int>> get_model_configs() const;

    void register_sample(int sample_idx,
                         const int64_t* input_ids,
                         const int64_t* attention_mask,
                         const int64_t* token_type_ids,
                         int actual_seq_len);
    void clear_samples();
    void stage_samples();

    static int get_bucket_index(int seq_len);
    static int get_bucket_seq_len(int bucket_idx);

    // Offline: batched
    void submit_batch(int bucket_idx,
                      const std::vector<uint64_t>& query_ids,
                      const std::vector<int>& sample_indices);

    // Server: direct batch=1
    void issue_query_direct(uint64_t query_id, int sample_idx);
    void issue_queries(const std::vector<uint64_t>& query_ids,
                       const std::vector<int>& sample_indices);

    void wait_all();
    void reset_counters();
    uint64_t get_completed_count() const { return completed_count_.load(); }
    uint64_t get_issued_count() const { return issued_count_.load(); }

    void set_store_predictions(bool store) { store_predictions_ = store; }
    std::unordered_map<int, BertOptPrediction> get_predictions() const;
    void clear_predictions();

    void enable_direct_loadgen(bool enable) { use_direct_loadgen_.store(enable); }

private:
    std::string model_path_;
    std::string device_prefix_;
    std::unordered_map<std::string, std::string> compile_properties_;
    int nireq_per_bucket_;
    std::vector<std::string> target_devices_;
    bool server_mode_ = false;

    ov::Core core_;
    std::shared_ptr<ov::Model> base_model_;
    std::vector<std::unique_ptr<BertOptDieContext>> die_contexts_;

    std::string input_ids_name_;
    std::string attention_mask_name_;
    std::string token_type_ids_name_;
    std::string start_logits_name_;
    std::string end_logits_name_;
    bool single_output_ = false;
    size_t start_output_idx_ = 0;
    size_t end_output_idx_ = 1;

    static constexpr int SLOT_FREE = -1;
    std::unique_ptr<std::atomic<int>[]> all_slot_states_;
    size_t total_slots_ = 0;

    mutable std::shared_mutex sample_mutex_;
    std::unordered_map<int, BertSampleData> samples_;

    // Fast bucket lookup for Server mode (avoids lock on hot path)
    std::vector<int8_t> sample_bucket_cache_;
    int sample_bucket_cache_size_ = 0;

    BertStagedBucket staged_buckets_[NUM_SEQ_BUCKETS];
    bool samples_staged_ = false;

    std::atomic<size_t> die_round_robin_[NUM_SEQ_BUCKETS];

    bool loaded_ = false;
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};

    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, BertOptPrediction> predictions_;

    std::atomic<bool> use_direct_loadgen_{false};

    std::vector<std::string> discover_devices();
    ov::AnyMap build_compile_properties();
    void map_input_output_names();
    std::shared_ptr<ov::Model> reshape_model(int batch_size, int seq_length);
    void build_bucket_cache();

    BertOptInferContext* acquire_request(size_t die_idx, int bucket_idx);
    void release_request(BertOptInferContext* ctx);
    void on_inference_complete(BertOptInferContext* ctx);

    void copy_sample_to_tensor(int sample_idx, int bucket_seq_len,
                               int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                               int offset_in_batch);
    void copy_staged_sample_to_tensor(int bucket_idx, size_t staged_idx, int bucket_seq_len,
                                      int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                                      int offset_in_batch);
};

} // namespace mlperf_ov
