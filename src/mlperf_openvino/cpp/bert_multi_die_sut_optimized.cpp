/**
 * Optimized BERT SUT implementation with dynamic sequence length buckets.
 *
 * - Offline mode: batched inference for throughput
 * - Server mode: batch=1 direct inference for minimum latency
 */

#include "bert_multi_die_sut_optimized.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <regex>

namespace mlperf_ov {

// =============================================================================
// STATIC HELPERS
// =============================================================================

int BertOptimizedSUT::get_bucket_index(int seq_len) {
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        if (seq_len <= SEQ_BUCKETS[i]) {
            return i;
        }
    }
    return NUM_SEQ_BUCKETS - 1;
}

int BertOptimizedSUT::get_bucket_seq_len(int bucket_idx) {
    if (bucket_idx < 0 || bucket_idx >= NUM_SEQ_BUCKETS) {
        return SEQ_BUCKETS[NUM_SEQ_BUCKETS - 1];
    }
    return SEQ_BUCKETS[bucket_idx];
}

// =============================================================================
// CONSTRUCTOR / DESTRUCTOR
// =============================================================================

BertOptimizedSUT::BertOptimizedSUT(
    const std::string& model_path,
    const std::string& device_prefix,
    const std::unordered_map<std::string, std::string>& compile_properties,
    int nireq_per_bucket)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      compile_properties_(compile_properties),
      nireq_per_bucket_(nireq_per_bucket) {

    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        die_round_robin_[i].store(0);
        staged_buckets_[i].seq_length = SEQ_BUCKETS[i];
    }
}

BertOptimizedSUT::~BertOptimizedSUT() {
    wait_all();
}

// =============================================================================
// CONFIGURATION
// =============================================================================

void BertOptimizedSUT::set_target_devices(const std::vector<std::string>& devices) {
    target_devices_ = devices;
}

std::vector<std::string> BertOptimizedSUT::get_active_devices() const {
    std::vector<std::string> result;
    for (const auto& die : die_contexts_) {
        result.push_back(die->device_name);
    }
    return result;
}

std::vector<std::pair<int, int>> BertOptimizedSUT::get_model_configs() const {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        int batch = server_mode_ ? SERVER_BATCH_SIZE : OFFLINE_BATCH_SIZES[i];
        result.emplace_back(batch, SEQ_BUCKETS[i]);
    }
    return result;
}

// =============================================================================
// DEVICE DISCOVERY
// =============================================================================

std::vector<std::string> BertOptimizedSUT::discover_devices() {
    if (!target_devices_.empty()) {
        return target_devices_;
    }

    std::vector<std::string> devices;
    auto all = core_.get_available_devices();
    std::regex pattern(device_prefix_ + R"(\.(\d+))");

    for (const auto& dev : all) {
        if (dev.find(device_prefix_) != 0) continue;
        if (dev.find("Simulator") != std::string::npos) continue;
        if (dev.find("FuncSim") != std::string::npos) continue;
        if (std::regex_match(dev, pattern)) {
            devices.push_back(dev);
        }
    }

    std::sort(devices.begin(), devices.end());
    return devices;
}

ov::AnyMap BertOptimizedSUT::build_compile_properties() {
    ov::AnyMap props;
    for (const auto& [k, v] : compile_properties_) {
        try { props[k] = std::stoi(v); continue; } catch (...) {}
        std::string upper = v;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == "TRUE") { props[k] = true; continue; }
        if (upper == "FALSE") { props[k] = false; continue; }
        props[k] = v;
    }
    return props;
}

void BertOptimizedSUT::map_input_output_names() {
    auto inputs = base_model_->inputs();
    for (const auto& input : inputs) {
        std::string name = input.get_any_name();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("input_ids") != std::string::npos || lower.find("input-ids") != std::string::npos) {
            input_ids_name_ = name;
        } else if (lower.find("attention") != std::string::npos || lower.find("mask") != std::string::npos) {
            attention_mask_name_ = name;
        } else if (lower.find("token_type") != std::string::npos || lower.find("segment") != std::string::npos) {
            token_type_ids_name_ = name;
        }
    }

    if (input_ids_name_.empty() && inputs.size() >= 1) input_ids_name_ = inputs[0].get_any_name();
    if (attention_mask_name_.empty() && inputs.size() >= 2) attention_mask_name_ = inputs[1].get_any_name();
    if (token_type_ids_name_.empty() && inputs.size() >= 3) token_type_ids_name_ = inputs[2].get_any_name();

    auto outputs = base_model_->outputs();
    if (outputs.size() == 1) {
        single_output_ = true;
        start_logits_name_ = outputs[0].get_any_name();
        end_logits_name_ = start_logits_name_;
        start_output_idx_ = 0;
        end_output_idx_ = 0;
    } else {
        single_output_ = false;
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::string name = outputs[i].get_any_name();
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("start") != std::string::npos) {
                start_logits_name_ = name;
                start_output_idx_ = i;
            } else if (lower.find("end") != std::string::npos) {
                end_logits_name_ = name;
                end_output_idx_ = i;
            }
        }
        if (start_logits_name_.empty()) {
            start_logits_name_ = outputs[0].get_any_name();
            start_output_idx_ = 0;
        }
        if (end_logits_name_.empty() && outputs.size() > 1) {
            end_logits_name_ = outputs[1].get_any_name();
            end_output_idx_ = 1;
        }
    }
}

std::shared_ptr<ov::Model> BertOptimizedSUT::reshape_model(int batch_size, int seq_length) {
    auto model = base_model_->clone();
    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& input : model->inputs()) {
        shapes[input.get_any_name()] = ov::PartialShape{batch_size, seq_length};
    }
    model->reshape(shapes);
    return model;
}

// =============================================================================
// LOAD
// =============================================================================

void BertOptimizedSUT::load() {
    if (loaded_) return;

    auto devices = discover_devices();
    if (devices.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " devices found");
    }

    base_model_ = core_.read_model(model_path_);
    map_input_output_names();

    auto props = build_compile_properties();

    // Calculate total slots
    total_slots_ = devices.size() * NUM_SEQ_BUCKETS * nireq_per_bucket_;
    all_slot_states_ = std::make_unique<std::atomic<int>[]>(total_slots_);
    for (size_t i = 0; i < total_slots_; ++i) {
        all_slot_states_[i].store(SLOT_FREE);
    }

    size_t slot_offset = 0;

    for (size_t die_idx = 0; die_idx < devices.size(); ++die_idx) {
        auto die = std::make_unique<BertOptDieContext>();
        die->device_name = devices[die_idx];
        die->die_idx = die_idx;

        for (int bucket_idx = 0; bucket_idx < NUM_SEQ_BUCKETS; ++bucket_idx) {
            int batch_size = server_mode_ ? SERVER_BATCH_SIZE : OFFLINE_BATCH_SIZES[bucket_idx];
            int seq_length = SEQ_BUCKETS[bucket_idx];

            auto reshaped = reshape_model(batch_size, seq_length);
            auto compiled = core_.compile_model(reshaped, die->device_name, props);

            auto model_ctx = std::make_unique<BertBucketModelContext>();
            model_ctx->compiled_model = compiled;
            model_ctx->batch_size = batch_size;
            model_ctx->seq_length = seq_length;
            model_ctx->slot_states = &all_slot_states_[slot_offset];
            model_ctx->num_requests = nireq_per_bucket_;

            for (int r = 0; r < nireq_per_bucket_; ++r) {
                auto ctx = std::make_unique<BertOptInferContext>();
                ctx->request = compiled.create_infer_request();
                ctx->batch_size = batch_size;
                ctx->seq_length = seq_length;
                ctx->bucket_idx = bucket_idx;
                ctx->die_idx = die_idx;
                ctx->pool_id = r;
                ctx->sut = this;

                ov::Shape shape{static_cast<size_t>(batch_size), static_cast<size_t>(seq_length)};
                ctx->input_ids_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->attention_mask_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->token_type_ids_tensor = ov::Tensor(ov::element::i64, shape);

                ctx->request.set_tensor(input_ids_name_, ctx->input_ids_tensor);
                ctx->request.set_tensor(attention_mask_name_, ctx->attention_mask_tensor);
                ctx->request.set_tensor(token_type_ids_name_, ctx->token_type_ids_tensor);

                BertOptInferContext* ctx_ptr = ctx.get();
                ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                    ctx_ptr->sut->on_inference_complete(ctx_ptr);
                });

                model_ctx->requests.push_back(std::move(ctx));
            }

            slot_offset += nireq_per_bucket_;
            die->bucket_models[bucket_idx] = std::move(model_ctx);
        }

        die_contexts_.push_back(std::move(die));
    }

    loaded_ = true;
}

// =============================================================================
// SAMPLE MANAGEMENT
// =============================================================================

void BertOptimizedSUT::register_sample(int sample_idx,
                                        const int64_t* input_ids,
                                        const int64_t* attention_mask,
                                        const int64_t* token_type_ids,
                                        int actual_seq_len) {
    int bucket_idx = get_bucket_index(actual_seq_len);
    int bucket_seq_len = SEQ_BUCKETS[bucket_idx];

    // Copy data for safe access
    BertSampleData data;
    data.input_ids.resize(bucket_seq_len);
    data.attention_mask.resize(bucket_seq_len);
    data.token_type_ids.resize(bucket_seq_len);
    data.actual_seq_len = actual_seq_len;
    data.bucket_idx = bucket_idx;

    int copy_len = std::min(actual_seq_len, bucket_seq_len);
    std::memcpy(data.input_ids.data(), input_ids, copy_len * sizeof(int64_t));
    std::memcpy(data.attention_mask.data(), attention_mask, copy_len * sizeof(int64_t));
    std::memcpy(data.token_type_ids.data(), token_type_ids, copy_len * sizeof(int64_t));

    // Zero-pad if needed
    if (copy_len < bucket_seq_len) {
        std::memset(data.input_ids.data() + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
        std::memset(data.attention_mask.data() + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
        std::memset(data.token_type_ids.data() + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
    }

    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_[sample_idx] = std::move(data);
}

void BertOptimizedSUT::clear_samples() {
    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_.clear();
    samples_staged_ = false;

    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        staged_buckets_[i].input_ids.clear();
        staged_buckets_[i].attention_mask.clear();
        staged_buckets_[i].token_type_ids.clear();
        staged_buckets_[i].samples.clear();
        staged_buckets_[i].sample_to_index.clear();
        staged_buckets_[i].staged = false;
    }
}

void BertOptimizedSUT::stage_samples() {
    if (samples_staged_) return;

    std::shared_lock<std::shared_mutex> lock(sample_mutex_);

    // Count samples per bucket
    std::vector<size_t> bucket_counts(NUM_SEQ_BUCKETS, 0);
    for (const auto& [idx, data] : samples_) {
        bucket_counts[data.bucket_idx]++;
    }

    // Pre-allocate buffers
    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        size_t total_elements = bucket_counts[b] * SEQ_BUCKETS[b];
        staged_buckets_[b].input_ids.resize(total_elements);
        staged_buckets_[b].attention_mask.resize(total_elements);
        staged_buckets_[b].token_type_ids.resize(total_elements);
        staged_buckets_[b].samples.reserve(bucket_counts[b]);
    }

    // Copy samples into staged buffers
    std::vector<size_t> bucket_offsets(NUM_SEQ_BUCKETS, 0);

    for (const auto& [sample_idx, data] : samples_) {
        int b = data.bucket_idx;
        int seq_len = SEQ_BUCKETS[b];
        size_t offset = bucket_offsets[b];

        std::memcpy(staged_buckets_[b].input_ids.data() + offset,
                    data.input_ids.data(), seq_len * sizeof(int64_t));
        std::memcpy(staged_buckets_[b].attention_mask.data() + offset,
                    data.attention_mask.data(), seq_len * sizeof(int64_t));
        std::memcpy(staged_buckets_[b].token_type_ids.data() + offset,
                    data.token_type_ids.data(), seq_len * sizeof(int64_t));

        size_t staged_idx = staged_buckets_[b].samples.size();
        staged_buckets_[b].samples.push_back({sample_idx, data.actual_seq_len, offset});
        staged_buckets_[b].sample_to_index[sample_idx] = staged_idx;

        bucket_offsets[b] += seq_len;
    }

    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        staged_buckets_[b].staged = true;
    }

    samples_staged_ = true;
}

// =============================================================================
// REQUEST POOL
// =============================================================================

BertOptInferContext* BertOptimizedSUT::acquire_request(size_t die_idx, int bucket_idx) {
    auto& die = die_contexts_[die_idx];
    auto& model_ctx = die->bucket_models[bucket_idx];

    size_t n = model_ctx->num_requests;
    size_t hint = model_ctx->pool_hint.load(std::memory_order_relaxed);

    for (size_t attempts = 0; attempts < n * 2; ++attempts) {
        size_t idx = (hint + attempts) % n;
        int expected = SLOT_FREE;
        if (model_ctx->slot_states[idx].compare_exchange_weak(
                expected, static_cast<int>(idx),
                std::memory_order_acquire, std::memory_order_relaxed)) {
            model_ctx->pool_hint.store((idx + 1) % n, std::memory_order_relaxed);
            return model_ctx->requests[idx].get();
        }
    }

    // Spin wait for a slot
    int spin = 0;
    while (true) {
        for (size_t idx = 0; idx < n; ++idx) {
            int expected = SLOT_FREE;
            if (model_ctx->slot_states[idx].compare_exchange_weak(
                    expected, static_cast<int>(idx),
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return model_ctx->requests[idx].get();
            }
        }
        if (++spin < 100) {
            #if defined(__x86_64__)
            __builtin_ia32_pause();
            #endif
        } else if (spin < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spin = 0;
        }
    }
}

void BertOptimizedSUT::release_request(BertOptInferContext* ctx) {
    auto& die = die_contexts_[ctx->die_idx];
    auto& model_ctx = die->bucket_models[ctx->bucket_idx];
    model_ctx->slot_states[ctx->pool_id].store(SLOT_FREE, std::memory_order_release);
}

// =============================================================================
// DATA COPY
// =============================================================================

void BertOptimizedSUT::copy_sample_to_tensor(int sample_idx, int bucket_seq_len,
                                              int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                                              int offset) {
    size_t dst_offset = static_cast<size_t>(offset) * bucket_seq_len;

    std::shared_lock<std::shared_mutex> lock(sample_mutex_);
    auto it = samples_.find(sample_idx);
    if (it != samples_.end()) {
        const BertSampleData& data = it->second;
        std::memcpy(ids_ptr + dst_offset, data.input_ids.data(), bucket_seq_len * sizeof(int64_t));
        std::memcpy(mask_ptr + dst_offset, data.attention_mask.data(), bucket_seq_len * sizeof(int64_t));
        std::memcpy(type_ptr + dst_offset, data.token_type_ids.data(), bucket_seq_len * sizeof(int64_t));
    } else {
        std::memset(ids_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
        std::memset(mask_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
        std::memset(type_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
    }
}

void BertOptimizedSUT::copy_staged_sample_to_tensor(int bucket_idx, size_t staged_idx, int bucket_seq_len,
                                                     int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                                                     int offset) {
    const auto& bucket = staged_buckets_[bucket_idx];
    size_t src_offset = bucket.samples[staged_idx].buffer_offset;
    size_t dst_offset = static_cast<size_t>(offset) * bucket_seq_len;

    std::memcpy(ids_ptr + dst_offset, bucket.input_ids.data() + src_offset, bucket_seq_len * sizeof(int64_t));
    std::memcpy(mask_ptr + dst_offset, bucket.attention_mask.data() + src_offset, bucket_seq_len * sizeof(int64_t));
    std::memcpy(type_ptr + dst_offset, bucket.token_type_ids.data() + src_offset, bucket_seq_len * sizeof(int64_t));
}

// =============================================================================
// INFERENCE COMPLETE
// =============================================================================

void BertOptimizedSUT::on_inference_complete(BertOptInferContext* ctx) {
    int actual = ctx->actual_batch_size;
    int dummies = ctx->num_dummies;
    int real = actual - dummies;

    if (store_predictions_ && real > 0) {
        ov::Tensor start_tensor = ctx->request.get_output_tensor(start_output_idx_);
        ov::Tensor end_tensor = single_output_ ? start_tensor : ctx->request.get_output_tensor(end_output_idx_);

        const float* start_data = start_tensor.data<float>();
        const float* end_data = end_tensor.data<float>();

        size_t logits_per_sample = start_tensor.get_size() / actual;

        std::lock_guard<std::mutex> lock(predictions_mutex_);
        for (int i = 0; i < real; ++i) {
            int sample_idx = ctx->sample_indices[i];
            BertOptPrediction pred;

            if (single_output_ && start_tensor.get_shape().back() == 2) {
                pred.start_logits.resize(logits_per_sample / 2);
                pred.end_logits.resize(logits_per_sample / 2);
                const float* sample = start_data + i * logits_per_sample;
                for (size_t j = 0; j < logits_per_sample / 2; ++j) {
                    pred.start_logits[j] = sample[j * 2];
                    pred.end_logits[j] = sample[j * 2 + 1];
                }
            } else if (single_output_) {
                size_t half = logits_per_sample / 2;
                pred.start_logits.assign(start_data + i * logits_per_sample,
                                         start_data + i * logits_per_sample + half);
                pred.end_logits.assign(start_data + i * logits_per_sample + half,
                                       start_data + (i + 1) * logits_per_sample);
            } else {
                pred.start_logits.assign(start_data + i * logits_per_sample,
                                         start_data + (i + 1) * logits_per_sample);
                pred.end_logits.assign(end_data + i * logits_per_sample,
                                       end_data + (i + 1) * logits_per_sample);
            }
            predictions_[sample_idx] = std::move(pred);
        }
    }

    if (use_direct_loadgen_.load(std::memory_order_relaxed) && real > 0) {
        mlperf::QuerySampleResponse responses[BertOptInferContext::MAX_BATCH];
        for (int i = 0; i < real; ++i) {
            responses[i] = {ctx->query_ids[i], 0, 0};
        }
        mlperf::QuerySamplesComplete(responses, real);
    }

    completed_count_.fetch_add(real, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);

    release_request(ctx);
}

// =============================================================================
// OFFLINE MODE: BATCHED SUBMISSION
// =============================================================================

void BertOptimizedSUT::submit_batch(int bucket_idx,
                                     const std::vector<uint64_t>& query_ids,
                                     const std::vector<int>& sample_indices) {
    if (!loaded_ || bucket_idx < 0 || bucket_idx >= NUM_SEQ_BUCKETS) return;

    if (!samples_staged_) {
        stage_samples();
    }

    int batch_size = OFFLINE_BATCH_SIZES[bucket_idx];
    int seq_len = SEQ_BUCKETS[bucket_idx];
    const auto& bucket = staged_buckets_[bucket_idx];

    int n = static_cast<int>(query_ids.size());

    for (int i = 0; i < n; i += batch_size) {
        int actual = std::min(batch_size, n - i);
        int dummies = batch_size - actual;

        size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();

        BertOptInferContext* ctx = acquire_request(die_idx, bucket_idx);

        for (int j = 0; j < actual; ++j) {
            ctx->query_ids[j] = query_ids[i + j];
            ctx->sample_indices[j] = sample_indices[i + j];
        }
        ctx->actual_batch_size = batch_size;
        ctx->num_dummies = dummies;

        int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* type = ctx->token_type_ids_tensor.data<int64_t>();

        for (int j = 0; j < actual; ++j) {
            int sample_idx = sample_indices[i + j];
            auto it = bucket.sample_to_index.find(sample_idx);
            if (it != bucket.sample_to_index.end()) {
                copy_staged_sample_to_tensor(bucket_idx, it->second, seq_len, ids, mask, type, j);
            } else {
                copy_sample_to_tensor(sample_idx, seq_len, ids, mask, type, j);
            }
        }

        // Pad dummies with first sample
        if (dummies > 0 && actual > 0) {
            int first_sample_idx = sample_indices[i];
            auto it = bucket.sample_to_index.find(first_sample_idx);
            for (int j = actual; j < batch_size; ++j) {
                if (it != bucket.sample_to_index.end()) {
                    copy_staged_sample_to_tensor(bucket_idx, it->second, seq_len, ids, mask, type, j);
                }
            }
        }

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(actual, std::memory_order_relaxed);
    }
}

// =============================================================================
// SERVER MODE: DIRECT SINGLE-QUERY INFERENCE
// =============================================================================

void BertOptimizedSUT::issue_query_direct(uint64_t query_id, int sample_idx) {
    if (!loaded_) return;

    // 1. Get bucket for this sample
    int bucket_idx = NUM_SEQ_BUCKETS - 1;  // Default to largest
    int seq_len = SEQ_BUCKETS[bucket_idx];
    {
        std::shared_lock<std::shared_mutex> lock(sample_mutex_);
        auto it = samples_.find(sample_idx);
        if (it != samples_.end()) {
            bucket_idx = it->second.bucket_idx;
            seq_len = SEQ_BUCKETS[bucket_idx];
        }
    }

    // 2. Select die (per-bucket round-robin)
    size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();

    // 3. Acquire request from pool
    BertOptInferContext* ctx = acquire_request(die_idx, bucket_idx);

    // 4. Setup context
    ctx->query_ids[0] = query_id;
    ctx->sample_indices[0] = sample_idx;
    ctx->actual_batch_size = 1;
    ctx->num_dummies = 0;

    // 5. Copy data (batch=1, offset=0)
    int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
    int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
    int64_t* type = ctx->token_type_ids_tensor.data<int64_t>();

    copy_sample_to_tensor(sample_idx, seq_len, ids, mask, type, 0);

    // 6. Start async inference
    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(1, std::memory_order_relaxed);
}

void BertOptimizedSUT::issue_queries(const std::vector<uint64_t>& query_ids,
                                      const std::vector<int>& sample_indices) {
    for (size_t i = 0; i < query_ids.size(); ++i) {
        issue_query_direct(query_ids[i], sample_indices[i]);
    }
}

// =============================================================================
// WAIT / STATS
// =============================================================================

void BertOptimizedSUT::wait_all() {
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BertOptimizedSUT::reset_counters() {
    wait_all();
    issued_count_.store(0);
    completed_count_.store(0);
}

std::unordered_map<int, BertOptPrediction> BertOptimizedSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void BertOptimizedSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

} // namespace mlperf_ov
