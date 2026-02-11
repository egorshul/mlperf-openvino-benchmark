/**
 * C++ SUT for 3D UNET on multi-die accelerators.
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

#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

namespace mlperf_ov {

class UNet3DMultiDieCppSUT;

struct UNet3DDieContext {
    std::string device_name;
    ov::CompiledModel compiled_model;
    int optimal_nireq = 1;
    size_t request_start_idx = 0;
    size_t request_count = 0;
    std::atomic<size_t> pool_search_hint{0};
};

struct UNet3DInferContext {
    ov::InferRequest request;
    ov::Tensor input_tensor;
    std::string die_name;
    size_t pool_id = 0;
    UNet3DMultiDieCppSUT* sut = nullptr;

    static constexpr int MAX_BATCH = 16;
    uint64_t query_ids[MAX_BATCH];
    int sample_indices[MAX_BATCH];
    int actual_batch_size = 0;
    int num_dummies = 0;
};

class UNet3DMultiDieCppSUT {
public:
    UNet3DMultiDieCppSUT(const std::string& model_path,
                          const std::string& device_prefix,
                          int batch_size = 1,
                          const std::unordered_map<std::string, std::string>& compile_properties = {},
                          bool use_nhwc_input = false,
                          int nireq_multiplier = 4);

    ~UNet3DMultiDieCppSUT();

    void load();
    void warmup(int iterations = 2);
    bool is_loaded() const { return loaded_; }
    int get_num_dies() const { return static_cast<int>(die_contexts_.size()); }
    std::vector<std::string> get_active_devices() const;
    int get_batch_size() const { return batch_size_; }
    int get_total_requests() const;
    std::string get_input_name() const { return input_name_; }
    std::string get_output_name() const { return output_name_; }

    // Batch dispatch
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

    // Response callback
    using BatchResponseCallback = std::function<void(const std::vector<uint64_t>& query_ids)>;
    void set_batch_response_callback(BatchResponseCallback callback);

    // Direct LoadGen mode
    void enable_direct_loadgen(bool enable);

    // Sample data cache
    void register_sample_data(int sample_idx, const float* data, size_t size);
    void clear_sample_data();

    // Explicit batching
    void enable_explicit_batching(bool enable, int batch_size = 1, int timeout_us = 2000);
    bool is_explicit_batching_enabled() const { return false; }

    // Target devices
    void set_target_devices(const std::vector<std::string>& devices) { target_devices_ = devices; }

    void issue_queries_server_fast(const std::vector<uint64_t>&, const std::vector<int>&) {}
    void run_server_benchmark(size_t, size_t, const std::string&, const std::string&,
                              const std::string&, double = 0, int64_t = 0,
                              int64_t = 0, int64_t = 0, bool = false) {}

private:
    // Config
    std::string model_path_;
    std::string device_prefix_;
    int batch_size_;
    std::unordered_map<std::string, std::string> compile_properties_;
    bool use_nhwc_input_;
    int nireq_multiplier_;
    std::vector<std::string> target_devices_;

    // OpenVINO
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    std::vector<std::unique_ptr<UNet3DDieContext>> die_contexts_;
    std::vector<std::string> active_devices_;

    // Request pool
    static constexpr int MAX_REQUESTS = 256;
    std::vector<std::unique_ptr<UNet3DInferContext>> infer_contexts_;
    std::atomic<int> request_slots_[MAX_REQUESTS];
    std::atomic<size_t> pool_search_hint_{0};

    // Model info
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
    std::atomic<bool> output_type_logged_{false};
    std::atomic<uint64_t> issued_count_{0};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<int> pending_count_{0};
    std::atomic<uint64_t> queued_count_{0};

    // Predictions
    bool store_predictions_ = false;
    mutable std::mutex predictions_mutex_;
    std::unordered_map<int, std::vector<float>> predictions_;

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

    // Round-robin die selection
    std::atomic<size_t> next_die_{0};

    // Helpers
    std::vector<std::string> discover_dies();
    ov::AnyMap build_compile_properties();
    size_t acquire_request();
    void release_request(size_t id);
    size_t acquire_request_for_die(size_t die_idx);
    void on_inference_complete(UNet3DInferContext* ctx);
};

} // namespace mlperf_ov
