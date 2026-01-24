/**
 * C++ SUT implementation for BERT on multi-die accelerators.
 *
 * Architecture:
 * - IssueQuery: pushes to work queue (non-blocking)
 * - Per-die threads: pull from queue, copy data, start_async
 * - Async callbacks: call QuerySamplesComplete directly
 */

#include "bert_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <chrono>

#include <openvino/runtime/properties.hpp>

namespace mlperf_ov {

static constexpr int SLOT_FREE = -1;

BertMultiDieCppSUT::BertMultiDieCppSUT(const std::string& model_path,
                                       const std::string& device_prefix,
                                       int batch_size,
                                       const std::unordered_map<std::string, std::string>& compile_properties,
                                       int nireq_multiplier)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      batch_size_(batch_size),
      compile_properties_(compile_properties),
      nireq_multiplier_(nireq_multiplier) {

    for (int i = 0; i < MAX_REQUESTS; ++i) {
        request_slots_[i].store(SLOT_FREE, std::memory_order_relaxed);
    }
    for (int i = 0; i < WORK_QUEUE_SIZE; ++i) {
        work_queue_[i].valid.store(false, std::memory_order_relaxed);
    }
    for (int d = 0; d < MAX_DIES; ++d) {
        batch_heads_[d].store(0, std::memory_order_relaxed);
        batch_tails_[d].store(0, std::memory_order_relaxed);
        for (int i = 0; i < BATCH_QUEUE_SIZE; ++i) {
            batch_queues_[d][i].valid.store(false, std::memory_order_relaxed);
        }
    }
}

BertMultiDieCppSUT::~BertMultiDieCppSUT() {
    issue_running_.store(false, std::memory_order_release);
    batcher_running_.store(false, std::memory_order_release);

    if (batcher_thread_.joinable()) {
        batcher_thread_.join();
    }

    for (auto& thread : issue_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    issue_threads_.clear();

    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        batch_response_callback_ = nullptr;
    }
}

// =============================================================================
// EXPLICIT BATCHING CONFIGURATION
// =============================================================================

void BertMultiDieCppSUT::enable_explicit_batching(bool enable, int batch_size, int timeout_us) {
    use_explicit_batching_ = enable;
    explicit_batch_size_ = batch_size;
    batch_timeout_us_ = timeout_us;
}

// =============================================================================
// REQUEST POOL
// =============================================================================

size_t BertMultiDieCppSUT::acquire_request() {
    size_t num_requests = infer_contexts_.size();
    size_t hint = pool_search_hint_.load(std::memory_order_relaxed);

    for (size_t attempts = 0; attempts < num_requests * 2; ++attempts) {
        size_t idx = (hint + attempts) % num_requests;
        int expected = SLOT_FREE;
        if (request_slots_[idx].compare_exchange_weak(
                expected, static_cast<int>(idx),
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            pool_search_hint_.store((idx + 1) % num_requests, std::memory_order_relaxed);
            return idx;
        }
    }

    // Spin wait
    int spin_count = 0;
    while (true) {
        for (size_t idx = 0; idx < num_requests; ++idx) {
            int expected = SLOT_FREE;
            if (request_slots_[idx].compare_exchange_weak(
                    expected, static_cast<int>(idx),
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                pool_search_hint_.store((idx + 1) % num_requests, std::memory_order_relaxed);
                return idx;
            }
        }
        spin_count++;
        if (spin_count < 100) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        } else if (spin_count < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spin_count = 0;
        }
    }
}

void BertMultiDieCppSUT::release_request(size_t id) {
    request_slots_[id].store(SLOT_FREE, std::memory_order_release);
}

size_t BertMultiDieCppSUT::acquire_request_for_die(size_t die_idx) {
    BertDieContext* die = die_contexts_[die_idx].get();
    size_t start = die->request_start_idx;
    size_t count = die->request_count;
    size_t hint = die->pool_search_hint.load(std::memory_order_relaxed);

    for (size_t attempts = 0; attempts < count * 2; ++attempts) {
        size_t local_idx = (hint + attempts) % count;
        size_t global_idx = start + local_idx;
        int expected = SLOT_FREE;
        if (request_slots_[global_idx].compare_exchange_weak(
                expected, static_cast<int>(global_idx),
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            die->pool_search_hint.store((local_idx + 1) % count, std::memory_order_relaxed);
            return global_idx;
        }
    }

    // Spin wait
    int spin_count = 0;
    while (true) {
        for (size_t local_idx = 0; local_idx < count; ++local_idx) {
            size_t global_idx = start + local_idx;
            int expected = SLOT_FREE;
            if (request_slots_[global_idx].compare_exchange_weak(
                    expected, static_cast<int>(global_idx),
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                die->pool_search_hint.store((local_idx + 1) % count, std::memory_order_relaxed);
                return global_idx;
            }
        }
        spin_count++;
        if (spin_count < 100) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        } else if (spin_count < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spin_count = 0;
        }
    }
}

// =============================================================================
// BATCHER THREAD
// =============================================================================

void BertMultiDieCppSUT::batcher_thread_func() {
    using namespace std::chrono;

    uint64_t batch_query_ids[64];
    int batch_sample_indices[64];
    int batch_count = 0;
    auto batch_start = steady_clock::now();

    while (batcher_running_.load(std::memory_order_acquire)) {
        size_t tail = work_tail_.load(std::memory_order_relaxed);
        size_t idx = tail % WORK_QUEUE_SIZE;

        bool got_sample = false;
        if (work_queue_[idx].valid.load(std::memory_order_acquire)) {
            if (work_tail_.compare_exchange_weak(tail, tail + 1,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                batch_query_ids[batch_count] = work_queue_[idx].query_id;
                batch_sample_indices[batch_count] = work_queue_[idx].sample_idx;
                work_queue_[idx].valid.store(false, std::memory_order_release);
                batch_count++;
                got_sample = true;

                if (batch_count == 1) {
                    batch_start = steady_clock::now();
                }
            }
        }

        bool should_flush = false;
        int num_dummies = 0;

        if (batch_count >= explicit_batch_size_) {
            should_flush = true;
        } else if (batch_count > 0 && !got_sample) {
            auto elapsed = duration_cast<microseconds>(steady_clock::now() - batch_start).count();
            if (elapsed >= batch_timeout_us_) {
                should_flush = true;
                num_dummies = explicit_batch_size_ - batch_count;

                for (int i = batch_count; i < explicit_batch_size_; ++i) {
                    batch_query_ids[i] = batch_query_ids[0];
                    batch_sample_indices[i] = batch_sample_indices[0];
                }
                batch_count = explicit_batch_size_;
            }
        }

        if (should_flush && batch_count > 0) {
            size_t num_dies = die_contexts_.size();
            size_t die_idx = next_die_.fetch_add(1, std::memory_order_relaxed) % num_dies;

            size_t head = batch_heads_[die_idx].fetch_add(1, std::memory_order_acq_rel);
            size_t batch_idx = head % BATCH_QUEUE_SIZE;

            while (batch_queues_[die_idx][batch_idx].valid.load(std::memory_order_acquire)) {
                #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                #endif
            }

            for (int i = 0; i < batch_count; ++i) {
                batch_queues_[die_idx][batch_idx].query_ids[i] = batch_query_ids[i];
                batch_queues_[die_idx][batch_idx].sample_indices[i] = batch_sample_indices[i];
            }
            batch_queues_[die_idx][batch_idx].actual_size = batch_count;
            batch_queues_[die_idx][batch_idx].num_dummies = num_dummies;
            batch_queues_[die_idx][batch_idx].valid.store(true, std::memory_order_release);

            batch_count = 0;
        }

        if (!got_sample) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        }
    }

    // Flush remaining
    if (batch_count > 0) {
        size_t num_dies = die_contexts_.size();
        size_t die_idx = next_die_.fetch_add(1, std::memory_order_relaxed) % num_dies;

        size_t head = batch_heads_[die_idx].fetch_add(1, std::memory_order_acq_rel);
        size_t batch_idx = head % BATCH_QUEUE_SIZE;

        while (batch_queues_[die_idx][batch_idx].valid.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (int i = 0; i < batch_count; ++i) {
            batch_queues_[die_idx][batch_idx].query_ids[i] = batch_query_ids[i];
            batch_queues_[die_idx][batch_idx].sample_indices[i] = batch_sample_indices[i];
        }
        batch_queues_[die_idx][batch_idx].actual_size = batch_count;
        batch_queues_[die_idx][batch_idx].num_dummies = 0;
        batch_queues_[die_idx][batch_idx].valid.store(true, std::memory_order_release);
    }
}

// =============================================================================
// ISSUE THREAD - BATCHED MODE
// =============================================================================

void BertMultiDieCppSUT::issue_thread_batched_func(size_t die_idx) {
    int idle_spins = 0;

    while (issue_running_.load(std::memory_order_acquire)) {
        size_t tail = batch_tails_[die_idx].load(std::memory_order_relaxed);
        size_t idx = tail % BATCH_QUEUE_SIZE;

        if (!batch_queues_[die_idx][idx].valid.load(std::memory_order_acquire)) {
            idle_spins++;
            if (idle_spins < 64) {
                #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                #endif
            } else if (idle_spins < 256) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                idle_spins = 0;
            }
            continue;
        }

        batch_tails_[die_idx].store(tail + 1, std::memory_order_relaxed);

        idle_spins = 0;
        int actual_size = batch_queues_[die_idx][idx].actual_size;
        int num_dummies = batch_queues_[die_idx][idx].num_dummies;
        int real_samples = actual_size - num_dummies;

        size_t req_id = acquire_request_for_die(die_idx);
        BertMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        for (int i = 0; i < actual_size; ++i) {
            ctx->query_ids[i] = batch_queues_[die_idx][idx].query_ids[i];
            ctx->sample_indices[i] = batch_queues_[die_idx][idx].sample_indices[i];
        }
        ctx->actual_batch_size = actual_size;
        ctx->num_dummies = num_dummies;

        batch_queues_[die_idx][idx].valid.store(false, std::memory_order_release);

        // Copy sample data for entire batch (3 tensors per sample)
        int64_t* ids_data = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask_data = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* type_data = ctx->token_type_ids_tensor.data<int64_t>();

        for (int i = 0; i < actual_size; ++i) {
            int sample_idx = ctx->sample_indices[i];

            const BertSampleData* sample = nullptr;
            {
                std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
                auto it = sample_data_cache_.find(sample_idx);
                if (it != sample_data_cache_.end()) {
                    sample = &it->second;
                }
            }

            if (sample) {
                size_t copy_size = sample->seq_length * sizeof(int64_t);
                std::memcpy(ids_data + i * seq_length_, sample->input_ids, copy_size);
                std::memcpy(mask_data + i * seq_length_, sample->attention_mask, copy_size);
                std::memcpy(type_data + i * seq_length_, sample->token_type_ids, copy_size);
            }
        }

        queued_count_.fetch_sub(real_samples, std::memory_order_relaxed);
        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(real_samples, std::memory_order_relaxed);
    }

    // Drain remaining
    while (true) {
        size_t tail = batch_tails_[die_idx].load(std::memory_order_relaxed);
        size_t idx = tail % BATCH_QUEUE_SIZE;

        if (!batch_queues_[die_idx][idx].valid.load(std::memory_order_acquire)) {
            break;
        }

        batch_tails_[die_idx].store(tail + 1, std::memory_order_relaxed);

        int actual_size = batch_queues_[die_idx][idx].actual_size;
        int num_dummies = batch_queues_[die_idx][idx].num_dummies;
        int real_samples = actual_size - num_dummies;

        size_t req_id = acquire_request_for_die(die_idx);
        BertMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        for (int i = 0; i < actual_size; ++i) {
            ctx->query_ids[i] = batch_queues_[die_idx][idx].query_ids[i];
            ctx->sample_indices[i] = batch_queues_[die_idx][idx].sample_indices[i];
        }
        ctx->actual_batch_size = actual_size;
        ctx->num_dummies = num_dummies;

        batch_queues_[die_idx][idx].valid.store(false, std::memory_order_release);

        int64_t* ids_data = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask_data = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* type_data = ctx->token_type_ids_tensor.data<int64_t>();

        for (int i = 0; i < actual_size; ++i) {
            int sample_idx = ctx->sample_indices[i];

            const BertSampleData* sample = nullptr;
            {
                std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
                auto it = sample_data_cache_.find(sample_idx);
                if (it != sample_data_cache_.end()) {
                    sample = &it->second;
                }
            }

            if (sample) {
                size_t copy_size = sample->seq_length * sizeof(int64_t);
                std::memcpy(ids_data + i * seq_length_, sample->input_ids, copy_size);
                std::memcpy(mask_data + i * seq_length_, sample->attention_mask, copy_size);
                std::memcpy(type_data + i * seq_length_, sample->token_type_ids, copy_size);
            }
        }

        queued_count_.fetch_sub(real_samples, std::memory_order_relaxed);
        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(real_samples, std::memory_order_relaxed);
    }
}

// =============================================================================
// ISSUE THREAD - SINGLE SAMPLE MODE
// =============================================================================

void BertMultiDieCppSUT::issue_thread_func(size_t die_idx) {
    int idle_spins = 0;

    while (issue_running_.load(std::memory_order_acquire)) {
        size_t tail = work_tail_.load(std::memory_order_relaxed);
        size_t idx = tail % WORK_QUEUE_SIZE;

        if (!work_queue_[idx].valid.load(std::memory_order_acquire)) {
            idle_spins++;
            if (idle_spins < 64) {
                #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                #endif
            } else if (idle_spins < 256) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                idle_spins = 0;
            }
            continue;
        }

        if (!work_tail_.compare_exchange_weak(tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }

        idle_spins = 0;
        uint64_t query_id = work_queue_[idx].query_id;
        int sample_idx = work_queue_[idx].sample_idx;
        work_queue_[idx].valid.store(false, std::memory_order_release);

        // Find sample data
        const BertSampleData* sample = nullptr;
        {
            std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
            auto it = sample_data_cache_.find(sample_idx);
            if (it != sample_data_cache_.end()) {
                sample = &it->second;
            }
        }

        if (!sample) {
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        size_t req_id = acquire_request_for_die(die_idx);
        BertMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;
        ctx->num_dummies = 0;

        // Copy 3 input tensors
        size_t copy_size = sample->seq_length * sizeof(int64_t);
        std::memcpy(ctx->input_ids_tensor.data<int64_t>(), sample->input_ids, copy_size);
        std::memcpy(ctx->attention_mask_tensor.data<int64_t>(), sample->attention_mask, copy_size);
        std::memcpy(ctx->token_type_ids_tensor.data<int64_t>(), sample->token_type_ids, copy_size);

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
        queued_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    // Drain on shutdown
    while (true) {
        size_t tail = work_tail_.load(std::memory_order_relaxed);
        size_t idx = tail % WORK_QUEUE_SIZE;

        if (!work_queue_[idx].valid.load(std::memory_order_acquire)) {
            break;
        }

        if (!work_tail_.compare_exchange_weak(tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }

        uint64_t query_id = work_queue_[idx].query_id;
        int sample_idx = work_queue_[idx].sample_idx;
        work_queue_[idx].valid.store(false, std::memory_order_release);

        const BertSampleData* sample = nullptr;
        {
            std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
            auto it = sample_data_cache_.find(sample_idx);
            if (it != sample_data_cache_.end()) {
                sample = &it->second;
            }
        }

        if (!sample) {
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        size_t req_id = acquire_request_for_die(die_idx);
        BertMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;
        ctx->num_dummies = 0;

        size_t copy_size = sample->seq_length * sizeof(int64_t);
        std::memcpy(ctx->input_ids_tensor.data<int64_t>(), sample->input_ids, copy_size);
        std::memcpy(ctx->attention_mask_tensor.data<int64_t>(), sample->attention_mask, copy_size);
        std::memcpy(ctx->token_type_ids_tensor.data<int64_t>(), sample->token_type_ids, copy_size);

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
        queued_count_.fetch_sub(1, std::memory_order_relaxed);
    }
}

// =============================================================================
// INFERENCE COMPLETE CALLBACK
// =============================================================================

void BertMultiDieCppSUT::on_inference_complete(BertMultiDieInferContext* ctx) {
    int actual_batch_size = ctx->actual_batch_size;
    int num_dummies = ctx->num_dummies;
    int real_samples = actual_batch_size - num_dummies;

    // Store predictions if needed
    if (store_predictions_ && real_samples > 0) {
        const float* start_data = nullptr;
        const float* end_data = nullptr;
        size_t logits_size = 0;

        if (single_output_) {
            // Single combined output
            ov::Tensor output = ctx->request.get_output_tensor(start_output_idx_);
            const float* data = output.data<float>();
            size_t total_size = output.get_size();

            if (output.get_shape().back() == 2) {
                // Shape: [batch, seq_len, 2] - interleaved
                logits_size = total_size / (actual_batch_size * 2);
            } else {
                // Shape: [batch, 2*seq_len] - concatenated
                logits_size = total_size / (actual_batch_size * 2);
            }

            std::lock_guard<std::mutex> lock(predictions_mutex_);
            for (int i = 0; i < real_samples; ++i) {
                int sample_idx = ctx->sample_indices[i];
                BertMultiDiePrediction pred;

                if (output.get_shape().back() == 2) {
                    // Interleaved
                    pred.start_logits.resize(logits_size);
                    pred.end_logits.resize(logits_size);
                    const float* sample_data = data + i * logits_size * 2;
                    for (size_t j = 0; j < logits_size; ++j) {
                        pred.start_logits[j] = sample_data[j * 2];
                        pred.end_logits[j] = sample_data[j * 2 + 1];
                    }
                } else {
                    // Concatenated
                    const float* sample_data = data + i * logits_size * 2;
                    pred.start_logits.assign(sample_data, sample_data + logits_size);
                    pred.end_logits.assign(sample_data + logits_size, sample_data + logits_size * 2);
                }
                predictions_[sample_idx] = std::move(pred);
            }
        } else {
            // Two separate outputs
            ov::Tensor start_tensor = ctx->request.get_output_tensor(start_output_idx_);
            ov::Tensor end_tensor = ctx->request.get_output_tensor(end_output_idx_);

            start_data = start_tensor.data<float>();
            end_data = end_tensor.data<float>();
            logits_size = start_tensor.get_size() / actual_batch_size;

            std::lock_guard<std::mutex> lock(predictions_mutex_);
            for (int i = 0; i < real_samples; ++i) {
                int sample_idx = ctx->sample_indices[i];
                BertMultiDiePrediction pred;
                pred.start_logits.assign(start_data + i * logits_size,
                                         start_data + (i + 1) * logits_size);
                pred.end_logits.assign(end_data + i * logits_size,
                                       end_data + (i + 1) * logits_size);
                predictions_[sample_idx] = std::move(pred);
            }
        }
    }

    // Prepare responses
    mlperf::QuerySampleResponse responses[BertMultiDieInferContext::MAX_BATCH];
    for (int i = 0; i < real_samples; ++i) {
        responses[i] = {ctx->query_ids[i], 0, 0};
    }

    size_t pool_id = ctx->pool_id;
    completed_count_.fetch_add(real_samples, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);
    release_request(pool_id);

    if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
        if (real_samples > 0) {
            mlperf::QuerySamplesComplete(responses, real_samples);
        }
    } else {
        std::vector<uint64_t> ids;
        ids.reserve(real_samples);
        for (int i = 0; i < real_samples; ++i) {
            ids.push_back(ctx->query_ids[i]);
        }
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (batch_response_callback_) {
            batch_response_callback_(ids);
        }
    }
}

// =============================================================================
// DIE DISCOVERY
// =============================================================================

std::vector<std::string> BertMultiDieCppSUT::discover_dies() {
    if (!target_devices_.empty()) {
        return target_devices_;
    }

    std::vector<std::string> dies;
    auto all_devices = core_.get_available_devices();
    std::regex die_pattern(device_prefix_ + R"(\.(\d+))");

    for (const auto& device : all_devices) {
        if (device.rfind(device_prefix_, 0) != 0) continue;
        if (device.find("Simulator") != std::string::npos) continue;
        if (device.find("FuncSim") != std::string::npos) continue;
        if (std::regex_match(device, die_pattern)) {
            dies.push_back(device);
        }
    }

    std::sort(dies.begin(), dies.end(), [](const std::string& a, const std::string& b) {
        auto get_num = [](const std::string& s) -> int {
            size_t pos = s.find('.');
            return (pos != std::string::npos) ? std::stoi(s.substr(pos + 1)) : 0;
        };
        return get_num(a) < get_num(b);
    });

    return dies;
}

ov::AnyMap BertMultiDieCppSUT::build_compile_properties() {
    ov::AnyMap properties;
    for (const auto& [key, value] : compile_properties_) {
        if (key == "AUTO_BATCH_TIMEOUT") {
            try {
                unsigned int timeout_ms = static_cast<unsigned int>(std::stoul(value));
                properties[ov::auto_batch_timeout.name()] = timeout_ms;
                continue;
            } catch (...) {
                std::cerr << "[WARN] Invalid AUTO_BATCH_TIMEOUT: " << value << std::endl;
            }
        }
        if (key == "OPTIMAL_BATCH_SIZE") {
            try {
                unsigned int batch_size = static_cast<unsigned int>(std::stoul(value));
                properties[ov::optimal_batch_size.name()] = batch_size;
                continue;
            } catch (...) {
                std::cerr << "[WARN] Invalid OPTIMAL_BATCH_SIZE: " << value << std::endl;
            }
        }
        try { properties[key] = std::stoi(value); continue; } catch (...) {}
        std::string upper = value;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == "TRUE") { properties[key] = true; continue; }
        if (upper == "FALSE") { properties[key] = false; continue; }
        properties[key] = value;
    }
    return properties;
}

void BertMultiDieCppSUT::map_input_names() {
    const auto& inputs = model_->inputs();

    for (const auto& input : inputs) {
        std::string name = input.get_any_name();
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (name_lower.find("input_id") != std::string::npos ||
            name_lower.find("input.1") != std::string::npos) {
            input_ids_name_ = name;
        } else if (name_lower.find("attention") != std::string::npos ||
                   name_lower.find("mask") != std::string::npos ||
                   name_lower.find("input.2") != std::string::npos) {
            attention_mask_name_ = name;
        } else if (name_lower.find("token_type") != std::string::npos ||
                   name_lower.find("segment") != std::string::npos ||
                   name_lower.find("input.3") != std::string::npos) {
            token_type_ids_name_ = name;
        }
    }

    // Fallback
    if (input_ids_name_.empty() && inputs.size() >= 1) {
        input_ids_name_ = inputs[0].get_any_name();
    }
    if (attention_mask_name_.empty() && inputs.size() >= 2) {
        attention_mask_name_ = inputs[1].get_any_name();
    }
    if (token_type_ids_name_.empty() && inputs.size() >= 3) {
        token_type_ids_name_ = inputs[2].get_any_name();
    }
}

void BertMultiDieCppSUT::map_output_names() {
    const auto& outputs = model_->outputs();

    if (outputs.size() == 1) {
        single_output_ = true;
        start_logits_name_ = outputs[0].get_any_name();
        end_logits_name_ = outputs[0].get_any_name();
        start_output_idx_ = 0;
        end_output_idx_ = 0;
    } else {
        single_output_ = false;

        for (size_t i = 0; i < outputs.size(); ++i) {
            std::string name = outputs[i].get_any_name();
            std::string name_lower = name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

            if (name_lower.find("start") != std::string::npos) {
                start_logits_name_ = name;
                start_output_idx_ = i;
            } else if (name_lower.find("end") != std::string::npos) {
                end_logits_name_ = name;
                end_output_idx_ = i;
            }
        }

        // Fallback
        if (start_logits_name_.empty() && outputs.size() >= 1) {
            start_logits_name_ = outputs[0].get_any_name();
            start_output_idx_ = 0;
        }
        if (end_logits_name_.empty() && outputs.size() >= 2) {
            end_logits_name_ = outputs[1].get_any_name();
            end_output_idx_ = 1;
        }
    }
}

// =============================================================================
// LOAD MODEL
// =============================================================================

void BertMultiDieCppSUT::load() {
    if (loaded_) return;

    active_devices_ = discover_dies();
    if (active_devices_.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " dies found");
    }

    if (active_devices_.size() > MAX_DIES) {
        std::cerr << "[WARN] Found " << active_devices_.size() << " dies, limiting to " << MAX_DIES << std::endl;
        active_devices_.resize(MAX_DIES);
    }

    model_ = core_.read_model(model_path_);

    // Map input/output names
    map_input_names();
    map_output_names();

    // Get sequence length from input shape
    const auto& inputs = model_->inputs();
    for (const auto& input : inputs) {
        if (input.get_any_name() == input_ids_name_) {
            auto shape = input.get_partial_shape();
            if (shape.rank().get_length() >= 2 && shape[1].is_static()) {
                seq_length_ = static_cast<int>(shape[1].get_length());
            }
            input_shape_ = shape.get_min_shape();
            input_type_ = input.get_element_type();
            break;
        }
    }

    // Determine actual batch size for model compilation
    int compile_batch_size = batch_size_;
    if (use_explicit_batching_ && explicit_batch_size_ > 1) {
        compile_batch_size = explicit_batch_size_;
    }

    // Reshape model for batch size
    if (compile_batch_size > 1 || input_shape_[0] == 0) {
        std::map<std::string, ov::PartialShape> new_shapes;
        for (const auto& input : inputs) {
            ov::PartialShape new_shape = input.get_partial_shape();
            new_shape[0] = compile_batch_size;
            new_shapes[input.get_any_name()] = new_shape;
        }
        model_->reshape(new_shapes);
        input_shape_ = model_->input(input_ids_name_).get_partial_shape().get_min_shape();
    }

    if (use_explicit_batching_ && explicit_batch_size_ > 1) {
        batch_size_ = explicit_batch_size_;
    }

    input_byte_size_ = seq_length_ * sizeof(int64_t);

    ov::AnyMap properties = build_compile_properties();

    size_t total_requests = 0;
    for (const auto& device_name : active_devices_) {
        auto die_ctx = std::make_unique<BertDieContext>();
        die_ctx->device_name = device_name;
        die_ctx->compiled_model = core_.compile_model(model_, device_name, properties);

        try {
            die_ctx->optimal_nireq = die_ctx->compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (...) {
            die_ctx->optimal_nireq = 4;
        }

        int num_requests = std::max(die_ctx->optimal_nireq * nireq_multiplier_, nireq_multiplier_ * 2);
        die_ctx->request_start_idx = total_requests;

        int actual_requests = 0;
        for (int i = 0; i < num_requests && total_requests < MAX_REQUESTS; ++i) {
            auto ctx = std::make_unique<BertMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = total_requests;
            ctx->sut = this;

            // Pre-allocate input tensors (int64)
            ov::Shape tensor_shape = {static_cast<size_t>(batch_size_), static_cast<size_t>(seq_length_)};

            ctx->input_ids_tensor = ov::Tensor(ov::element::i64, tensor_shape);
            ctx->attention_mask_tensor = ov::Tensor(ov::element::i64, tensor_shape);
            ctx->token_type_ids_tensor = ov::Tensor(ov::element::i64, tensor_shape);

            ctx->request.set_tensor(input_ids_name_, ctx->input_ids_tensor);
            ctx->request.set_tensor(attention_mask_name_, ctx->attention_mask_tensor);
            ctx->request.set_tensor(token_type_ids_name_, ctx->token_type_ids_tensor);

            BertMultiDieInferContext* ctx_ptr = ctx.get();
            ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                ctx_ptr->sut->on_inference_complete(ctx_ptr);
            });

            infer_contexts_.push_back(std::move(ctx));
            request_slots_[total_requests].store(SLOT_FREE, std::memory_order_relaxed);
            total_requests++;
            actual_requests++;
        }

        die_ctx->request_count = actual_requests;
        die_contexts_.push_back(std::move(die_ctx));
    }

    // Calculate output size
    auto output_shape = model_->outputs()[start_output_idx_].get_partial_shape().get_min_shape();
    single_output_size_ = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        single_output_size_ *= output_shape[i];
    }

    // Start threads
    issue_running_.store(true, std::memory_order_release);

    if (use_explicit_batching_) {
        batcher_running_.store(true, std::memory_order_release);
        batcher_thread_ = std::thread(&BertMultiDieCppSUT::batcher_thread_func, this);

        for (size_t die_idx = 0; die_idx < die_contexts_.size(); ++die_idx) {
            issue_threads_.emplace_back(&BertMultiDieCppSUT::issue_thread_batched_func, this, die_idx);
        }
    } else {
        for (size_t die_idx = 0; die_idx < die_contexts_.size(); ++die_idx) {
            issue_threads_.emplace_back(&BertMultiDieCppSUT::issue_thread_func, this, die_idx);
        }
    }

    std::cout << "[BERT SUT] Loaded: " << die_contexts_.size() << " dies, "
              << total_requests << " requests, seq_len=" << seq_length_;
    if (use_explicit_batching_) {
        std::cout << ", batch=" << batch_size_ << ", timeout=" << batch_timeout_us_ << "us";
    }
    std::cout << std::endl;

    loaded_ = true;
}

// =============================================================================
// PUBLIC API
// =============================================================================

std::vector<std::string> BertMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

int BertMultiDieCppSUT::get_total_requests() const {
    return static_cast<int>(infer_contexts_.size());
}

void BertMultiDieCppSUT::start_async_batch(const int64_t* input_ids,
                                            const int64_t* attention_mask,
                                            const int64_t* token_type_ids,
                                            int seq_length,
                                            const std::vector<uint64_t>& query_ids,
                                            const std::vector<int>& sample_indices,
                                            int actual_batch_size) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t id = acquire_request();
    BertMultiDieInferContext* ctx = infer_contexts_[id].get();

    actual_batch_size = std::min(actual_batch_size, BertMultiDieInferContext::MAX_BATCH);
    for (int i = 0; i < actual_batch_size; ++i) {
        ctx->query_ids[i] = query_ids[i];
        ctx->sample_indices[i] = sample_indices[i];
    }
    ctx->actual_batch_size = actual_batch_size;
    ctx->num_dummies = 0;

    // Copy data
    size_t copy_size = actual_batch_size * seq_length * sizeof(int64_t);
    std::memcpy(ctx->input_ids_tensor.data<int64_t>(), input_ids, copy_size);
    std::memcpy(ctx->attention_mask_tensor.data<int64_t>(), attention_mask, copy_size);
    std::memcpy(ctx->token_type_ids_tensor.data<int64_t>(), token_type_ids, copy_size);

    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
}

void BertMultiDieCppSUT::enable_direct_loadgen(bool enable) {
    use_direct_loadgen_.store(enable, std::memory_order_release);
}

void BertMultiDieCppSUT::wait_all() {
    while (queued_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BertMultiDieCppSUT::reset_counters() {
    wait_all();
    issued_count_.store(0, std::memory_order_relaxed);
    completed_count_.store(0, std::memory_order_relaxed);
    queued_count_.store(0, std::memory_order_relaxed);
}

std::unordered_map<int, BertMultiDiePrediction> BertMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void BertMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void BertMultiDieCppSUT::set_batch_response_callback(BatchResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_response_callback_ = callback;
}

void BertMultiDieCppSUT::register_sample_data(int sample_idx,
                                               const int64_t* input_ids,
                                               const int64_t* attention_mask,
                                               const int64_t* token_type_ids,
                                               size_t seq_length) {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_[sample_idx] = {input_ids, attention_mask, token_type_ids, seq_length};
}

void BertMultiDieCppSUT::clear_sample_data() {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_.clear();
}

void BertMultiDieCppSUT::issue_queries_server_fast(
    const std::vector<uint64_t>& query_ids,
    const std::vector<int>& sample_indices) {

    size_t num_samples = query_ids.size();
    for (size_t i = 0; i < num_samples; ++i) {
        size_t head = work_head_.fetch_add(1, std::memory_order_acq_rel);
        size_t idx = head % WORK_QUEUE_SIZE;

        while (work_queue_[idx].valid.load(std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        }

        work_queue_[idx].query_id = query_ids[i];
        work_queue_[idx].sample_idx = sample_indices[i];
        work_queue_[idx].valid.store(true, std::memory_order_release);
        queued_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

// =============================================================================
// SERVER BENCHMARK
// =============================================================================

void BertMultiDieCppSUT::run_server_benchmark(
    size_t total_sample_count,
    size_t performance_sample_count,
    const std::string& mlperf_conf_path,
    const std::string& user_conf_path,
    const std::string& log_output_dir,
    double target_qps,
    int64_t target_latency_ns,
    int64_t min_duration_ms,
    int64_t min_query_count,
    bool is_accuracy_mode) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    BertServerQSL server_qsl(total_sample_count, performance_sample_count);
    BertServerSUT server_sut(this);

    mlperf::TestSettings test_settings;
    test_settings.scenario = mlperf::TestScenario::Server;
    test_settings.mode = is_accuracy_mode ? mlperf::TestMode::AccuracyOnly : mlperf::TestMode::PerformanceOnly;

    if (!mlperf_conf_path.empty()) {
        test_settings.FromConfig(mlperf_conf_path, "bert", "Server");
    }
    if (!user_conf_path.empty()) {
        test_settings.FromConfig(user_conf_path, "bert", "Server");
    }

    if (target_qps > 0) {
        test_settings.server_target_qps = target_qps;
    }
    if (target_latency_ns > 0) {
        test_settings.server_target_latency_ns = target_latency_ns;
    }
    if (min_duration_ms > 0) {
        test_settings.min_duration_ms = static_cast<uint64_t>(min_duration_ms);
    }
    if (min_query_count > 0) {
        test_settings.min_query_count = static_cast<uint64_t>(min_query_count);
    }

    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = log_output_dir;
    log_settings.log_output.copy_summary_to_stdout = true;

    std::string mode_str = is_accuracy_mode ? "accuracy" : "performance";
    std::cout << "[BERT Server] Starting " << mode_str << " run: "
              << "qps=" << test_settings.server_target_qps
              << ", latency=" << test_settings.server_target_latency_ns / 1e6 << "ms"
              << ", samples=" << (is_accuracy_mode ? total_sample_count : performance_sample_count)
              << std::endl;

    mlperf::StartTest(&server_sut, &server_qsl, test_settings, log_settings);

    std::cout << "[BERT Server] Completed: " << completed_count_.load() << " samples" << std::endl;
}

} // namespace mlperf_ov
