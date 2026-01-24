/**
 * Optimized BERT SUT implementation with dynamic sequence length buckets.
 */

#include "bert_multi_die_sut_optimized.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
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
    return NUM_SEQ_BUCKETS - 1;  // Fallback to largest
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
    int nireq_per_config)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      compile_properties_(compile_properties),
      nireq_per_config_(nireq_per_config) {

    // Default batch sizes per bucket
    bucket_batch_sizes_.assign(BATCH_SIZES, BATCH_SIZES + NUM_SEQ_BUCKETS);

    // Initialize queues
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        die_round_robin_[i].store(0);
        for (int j = 0; j < BATCH_QUEUE_SIZE; ++j) {
            batch_queues_[i].valid[j].store(false);
        }
    }
}

BertOptimizedSUT::~BertOptimizedSUT() {
    batcher_running_.store(false);
    dispatch_running_.store(false);

    if (batcher_thread_.joinable()) {
        batcher_thread_.join();
    }
    for (auto& t : dispatch_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    wait_all();
}

// =============================================================================
// CONFIGURATION
// =============================================================================

void BertOptimizedSUT::set_target_devices(const std::vector<std::string>& devices) {
    target_devices_ = devices;
}

void BertOptimizedSUT::set_bucket_batch_sizes(const std::vector<int>& batch_sizes) {
    if (batch_sizes.size() == NUM_SEQ_BUCKETS) {
        bucket_batch_sizes_ = batch_sizes;
    }
}

std::vector<std::string> BertOptimizedSUT::get_active_devices() const {
    std::vector<std::string> result;
    for (const auto& die : die_contexts_) {
        result.push_back(die->device_name);
    }
    return result;
}

int BertOptimizedSUT::get_num_model_configs() const {
    return NUM_SEQ_BUCKETS;  // One config per bucket
}

std::vector<std::pair<int, int>> BertOptimizedSUT::get_model_configs() const {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        result.emplace_back(bucket_batch_sizes_[i], SEQ_BUCKETS[i]);
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
    // Map inputs
    for (const auto& input : base_model_->inputs()) {
        std::string name = input.get_any_name();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower.find("input_id") != std::string::npos || lower.find("input.1") != std::string::npos) {
            input_ids_name_ = name;
        } else if (lower.find("attention") != std::string::npos || lower.find("mask") != std::string::npos) {
            attention_mask_name_ = name;
        } else if (lower.find("token_type") != std::string::npos || lower.find("segment") != std::string::npos) {
            token_type_ids_name_ = name;
        }
    }

    // Fallback
    auto inputs = base_model_->inputs();
    if (input_ids_name_.empty() && inputs.size() >= 1) input_ids_name_ = inputs[0].get_any_name();
    if (attention_mask_name_.empty() && inputs.size() >= 2) attention_mask_name_ = inputs[1].get_any_name();
    if (token_type_ids_name_.empty() && inputs.size() >= 3) token_type_ids_name_ = inputs[2].get_any_name();

    // Map outputs
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

    std::cout << "[BertOptimizedSUT] Found " << devices.size() << " devices" << std::endl;

    // Load base model
    base_model_ = core_.read_model(model_path_);
    map_input_output_names();

    auto props = build_compile_properties();

    // Calculate total slots needed
    size_t total_slots = devices.size() * NUM_SEQ_BUCKETS * nireq_per_config_;
    all_slot_states_.resize(total_slots);
    for (auto& s : all_slot_states_) {
        s.store(SLOT_FREE);
    }

    size_t slot_offset = 0;

    // Create die contexts
    for (size_t die_idx = 0; die_idx < devices.size(); ++die_idx) {
        auto die = std::make_unique<DieContext>();
        die->device_name = devices[die_idx];
        die->die_idx = die_idx;

        // Compile model for each bucket configuration
        for (int bucket_idx = 0; bucket_idx < NUM_SEQ_BUCKETS; ++bucket_idx) {
            int batch_size = bucket_batch_sizes_[bucket_idx];
            int seq_length = SEQ_BUCKETS[bucket_idx];

            BertModelConfig config{batch_size, seq_length};

            auto reshaped = reshape_model(batch_size, seq_length);
            auto compiled = core_.compile_model(reshaped, die->device_name, props);

            auto model_ctx = std::make_unique<ModelContext>();
            model_ctx->config = config;
            model_ctx->compiled_model = compiled;
            model_ctx->slot_states = &all_slot_states_[slot_offset];
            model_ctx->num_requests = nireq_per_config_;

            // Create inference requests
            for (int r = 0; r < nireq_per_config_; ++r) {
                auto ctx = std::make_unique<BertInferContext>();
                ctx->request = compiled.create_infer_request();
                ctx->config = config;
                ctx->die_idx = die_idx;
                ctx->pool_id = r;
                ctx->sut = this;

                // Pre-allocate tensors
                ov::Shape shape{static_cast<size_t>(batch_size), static_cast<size_t>(seq_length)};
                ctx->input_ids_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->attention_mask_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->token_type_ids_tensor = ov::Tensor(ov::element::i64, shape);

                ctx->request.set_tensor(input_ids_name_, ctx->input_ids_tensor);
                ctx->request.set_tensor(attention_mask_name_, ctx->attention_mask_tensor);
                ctx->request.set_tensor(token_type_ids_name_, ctx->token_type_ids_tensor);

                // Set callback
                BertInferContext* ctx_ptr = ctx.get();
                ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                    ctx_ptr->sut->on_inference_complete(ctx_ptr);
                });

                model_ctx->requests.push_back(std::move(ctx));
            }

            slot_offset += nireq_per_config_;
            die->models[config] = std::move(model_ctx);
        }

        die_contexts_.push_back(std::move(die));
    }

    // Start dispatch threads (one per bucket)
    dispatch_running_.store(true);
    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        dispatch_threads_.emplace_back(&BertOptimizedSUT::dispatch_thread_func, this, b);
    }

    // Start batcher thread for Server mode
    batcher_running_.store(true);
    batcher_thread_ = std::thread(&BertOptimizedSUT::batcher_thread_func, this);

    std::cout << "[BertOptimizedSUT] Loaded: " << devices.size() << " dies, "
              << NUM_SEQ_BUCKETS << " bucket configs, "
              << total_slots << " total requests" << std::endl;

    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        std::cout << "  Bucket " << i << ": seq=" << SEQ_BUCKETS[i]
                  << ", batch=" << bucket_batch_sizes_[i] << std::endl;
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

    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_[sample_idx] = {input_ids, attention_mask, token_type_ids, actual_seq_len, bucket_idx};
}

void BertOptimizedSUT::clear_samples() {
    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_.clear();
}

// =============================================================================
// REQUEST POOL
// =============================================================================

size_t BertOptimizedSUT::acquire_request(size_t die_idx, const BertModelConfig& config) {
    auto& die = die_contexts_[die_idx];
    auto it = die->models.find(config);
    if (it == die->models.end()) {
        throw std::runtime_error("Model config not found");
    }

    ModelContext* model_ctx = it->second.get();
    size_t n = model_ctx->num_requests;
    size_t hint = model_ctx->pool_hint.load(std::memory_order_relaxed);

    // Try to acquire
    for (size_t attempts = 0; attempts < n * 2; ++attempts) {
        size_t idx = (hint + attempts) % n;
        int expected = SLOT_FREE;
        if (model_ctx->slot_states[idx].compare_exchange_weak(
                expected, static_cast<int>(idx),
                std::memory_order_acquire, std::memory_order_relaxed)) {
            model_ctx->pool_hint.store((idx + 1) % n, std::memory_order_relaxed);
            return idx;
        }
    }

    // Spin wait
    int spin = 0;
    while (true) {
        for (size_t idx = 0; idx < n; ++idx) {
            int expected = SLOT_FREE;
            if (model_ctx->slot_states[idx].compare_exchange_weak(
                    expected, static_cast<int>(idx),
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return idx;
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

void BertOptimizedSUT::release_request(size_t die_idx, const BertModelConfig& config, size_t pool_id) {
    auto& die = die_contexts_[die_idx];
    auto it = die->models.find(config);
    if (it != die->models.end()) {
        it->second->slot_states[pool_id].store(SLOT_FREE, std::memory_order_release);
    }
}

// =============================================================================
// DATA COPY
// =============================================================================

void BertOptimizedSUT::copy_sample_to_tensor(int sample_idx, int bucket_seq_len,
                                              int64_t* ids_ptr, int64_t* mask_ptr, int64_t* type_ptr,
                                              int offset) {
    const BertSampleInfo* info = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(sample_mutex_);
        auto it = samples_.find(sample_idx);
        if (it != samples_.end()) {
            info = &it->second;
        }
    }

    size_t dst_offset = offset * bucket_seq_len;

    if (info) {
        // Copy actual data (up to actual_seq_len)
        int copy_len = std::min(info->actual_seq_len, bucket_seq_len);
        std::memcpy(ids_ptr + dst_offset, info->input_ids, copy_len * sizeof(int64_t));
        std::memcpy(mask_ptr + dst_offset, info->attention_mask, copy_len * sizeof(int64_t));
        std::memcpy(type_ptr + dst_offset, info->token_type_ids, copy_len * sizeof(int64_t));

        // Zero-pad if needed
        if (copy_len < bucket_seq_len) {
            std::memset(ids_ptr + dst_offset + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
            std::memset(mask_ptr + dst_offset + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
            std::memset(type_ptr + dst_offset + copy_len, 0, (bucket_seq_len - copy_len) * sizeof(int64_t));
        }
    } else {
        // Sample not found - zero fill
        std::memset(ids_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
        std::memset(mask_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
        std::memset(type_ptr + dst_offset, 0, bucket_seq_len * sizeof(int64_t));
    }
}

// =============================================================================
// INFERENCE COMPLETE
// =============================================================================

void BertOptimizedSUT::on_inference_complete(BertInferContext* ctx) {
    int actual = ctx->actual_batch_size;
    int dummies = ctx->num_dummies;
    int real = actual - dummies;

    // Store predictions if needed
    if (store_predictions_ && real > 0) {
        ov::Tensor start_tensor = ctx->request.get_output_tensor(start_output_idx_);
        ov::Tensor end_tensor = single_output_ ? start_tensor : ctx->request.get_output_tensor(end_output_idx_);

        const float* start_data = start_tensor.data<float>();
        const float* end_data = end_tensor.data<float>();

        size_t logits_per_sample = start_tensor.get_size() / actual;

        std::lock_guard<std::mutex> lock(predictions_mutex_);
        for (int i = 0; i < real; ++i) {
            int sample_idx = ctx->sample_indices[i];
            BertPrediction pred;

            if (single_output_ && start_tensor.get_shape().back() == 2) {
                // Interleaved format [batch, seq, 2]
                pred.start_logits.resize(logits_per_sample / 2);
                pred.end_logits.resize(logits_per_sample / 2);
                const float* sample = start_data + i * logits_per_sample;
                for (size_t j = 0; j < logits_per_sample / 2; ++j) {
                    pred.start_logits[j] = sample[j * 2];
                    pred.end_logits[j] = sample[j * 2 + 1];
                }
            } else if (single_output_) {
                // Concatenated format
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

    // Send responses
    if (use_direct_loadgen_.load(std::memory_order_relaxed) && real > 0) {
        mlperf::QuerySampleResponse responses[BertInferContext::MAX_BATCH];
        for (int i = 0; i < real; ++i) {
            responses[i] = {ctx->query_ids[i], 0, 0};
        }
        mlperf::QuerySamplesComplete(responses, real);
    }

    completed_count_.fetch_add(real, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);

    release_request(ctx->die_idx, ctx->config, ctx->pool_id);
}

// =============================================================================
// OFFLINE BATCH SUBMISSION
// =============================================================================

void BertOptimizedSUT::submit_batch(int bucket_idx,
                                     const std::vector<uint64_t>& query_ids,
                                     const std::vector<int>& sample_indices) {
    if (!loaded_ || bucket_idx < 0 || bucket_idx >= NUM_SEQ_BUCKETS) return;

    int batch_size = bucket_batch_sizes_[bucket_idx];
    int seq_len = SEQ_BUCKETS[bucket_idx];
    BertModelConfig config{batch_size, seq_len};

    int n = static_cast<int>(query_ids.size());

    // Process in batches
    for (int i = 0; i < n; i += batch_size) {
        int actual = std::min(batch_size, n - i);
        int dummies = batch_size - actual;

        // Round-robin die selection
        size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();

        size_t req_id = acquire_request(die_idx, config);
        auto& model_ctx = die_contexts_[die_idx]->models[config];
        BertInferContext* ctx = model_ctx->requests[req_id].get();

        // Fill context
        for (int j = 0; j < actual; ++j) {
            ctx->query_ids[j] = query_ids[i + j];
            ctx->sample_indices[j] = sample_indices[i + j];
        }
        ctx->actual_batch_size = batch_size;
        ctx->num_dummies = dummies;

        // Copy data
        int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* type = ctx->token_type_ids_tensor.data<int64_t>();

        for (int j = 0; j < actual; ++j) {
            copy_sample_to_tensor(sample_indices[i + j], seq_len, ids, mask, type, j);
        }

        // Pad with first sample for dummies
        for (int j = actual; j < batch_size; ++j) {
            copy_sample_to_tensor(sample_indices[i], seq_len, ids, mask, type, j);
        }

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(actual, std::memory_order_relaxed);
    }
}

// =============================================================================
// SERVER MODE: ISSUE QUERIES
// =============================================================================

void BertOptimizedSUT::issue_queries(const std::vector<uint64_t>& query_ids,
                                      const std::vector<int>& sample_indices) {
    // Push to per-bucket queues
    for (size_t i = 0; i < query_ids.size(); ++i) {
        int sample_idx = sample_indices[i];

        // Get bucket for this sample
        int bucket_idx = NUM_SEQ_BUCKETS - 1;  // Default
        {
            std::shared_lock<std::shared_mutex> lock(sample_mutex_);
            auto it = samples_.find(sample_idx);
            if (it != samples_.end()) {
                bucket_idx = it->second.bucket_idx;
            }
        }

        // Push to bucket queue
        auto& queue = bucket_queues_[bucket_idx];
        size_t head = queue.head.fetch_add(1, std::memory_order_acq_rel);
        size_t idx = head % QUEUE_SIZE;

        // Spin if slot is full (shouldn't happen often with large enough queue)
        while (queue.items[idx].query_id != 0 && queue.items[idx].sample_idx != -1) {
            std::this_thread::yield();
        }

        queue.items[idx].query_id = query_ids[i];
        queue.items[idx].sample_idx = sample_idx;
        queue.items[idx].bucket_idx = bucket_idx;
    }
}

// =============================================================================
// BATCHER THREAD
// =============================================================================

void BertOptimizedSUT::batcher_thread_func() {
    using namespace std::chrono;

    std::vector<BertWorkItem> pending[NUM_SEQ_BUCKETS];
    auto last_flush = steady_clock::now();

    while (batcher_running_.load(std::memory_order_acquire)) {
        bool got_any = false;

        // Collect from all bucket queues
        for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
            auto& queue = bucket_queues_[b];
            size_t tail = queue.tail.load(std::memory_order_relaxed);
            size_t idx = tail % QUEUE_SIZE;

            if (queue.items[idx].sample_idx >= 0) {
                if (queue.tail.compare_exchange_weak(tail, tail + 1)) {
                    pending[b].push_back(queue.items[idx]);
                    queue.items[idx].sample_idx = -1;
                    queue.items[idx].query_id = 0;
                    got_any = true;
                }
            }
        }

        // Check if we should flush any batches
        auto now = steady_clock::now();
        bool timeout = duration_cast<microseconds>(now - last_flush).count() >= batch_timeout_us_;

        for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
            int target = bucket_batch_sizes_[b];

            while (static_cast<int>(pending[b].size()) >= target) {
                // Create full batch
                BertBatch batch;
                batch.bucket_idx = b;
                batch.target_batch_size = target;
                batch.num_dummies = 0;

                for (int i = 0; i < target; ++i) {
                    batch.query_ids.push_back(pending[b][i].query_id);
                    batch.sample_indices.push_back(pending[b][i].sample_idx);
                }
                pending[b].erase(pending[b].begin(), pending[b].begin() + target);

                // Push to batch queue
                auto& bq = batch_queues_[b];
                size_t head = bq.head.fetch_add(1, std::memory_order_acq_rel);
                size_t slot = head % BATCH_QUEUE_SIZE;

                while (bq.valid[slot].load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }

                bq.batches[slot] = std::move(batch);
                bq.valid[slot].store(true, std::memory_order_release);

                last_flush = now;
            }

            // Flush partial batch on timeout
            if (timeout && !pending[b].empty() && static_cast<int>(pending[b].size()) >= min_batch_size_) {
                BertBatch batch;
                batch.bucket_idx = b;
                batch.target_batch_size = target;

                int actual = static_cast<int>(pending[b].size());
                batch.num_dummies = target - actual;

                for (const auto& item : pending[b]) {
                    batch.query_ids.push_back(item.query_id);
                    batch.sample_indices.push_back(item.sample_idx);
                }
                pending[b].clear();

                auto& bq = batch_queues_[b];
                size_t head = bq.head.fetch_add(1, std::memory_order_acq_rel);
                size_t slot = head % BATCH_QUEUE_SIZE;

                while (bq.valid[slot].load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }

                bq.batches[slot] = std::move(batch);
                bq.valid[slot].store(true, std::memory_order_release);

                last_flush = now;
            }
        }

        if (!got_any) {
            std::this_thread::sleep_for(microseconds(50));
        }
    }

    // Flush remaining
    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        while (!pending[b].empty()) {
            int target = bucket_batch_sizes_[b];
            int actual = std::min(target, static_cast<int>(pending[b].size()));

            BertBatch batch;
            batch.bucket_idx = b;
            batch.target_batch_size = target;
            batch.num_dummies = target - actual;

            for (int i = 0; i < actual; ++i) {
                batch.query_ids.push_back(pending[b][i].query_id);
                batch.sample_indices.push_back(pending[b][i].sample_idx);
            }
            pending[b].erase(pending[b].begin(), pending[b].begin() + actual);

            auto& bq = batch_queues_[b];
            size_t head = bq.head.fetch_add(1, std::memory_order_acq_rel);
            size_t slot = head % BATCH_QUEUE_SIZE;

            while (bq.valid[slot].load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            bq.batches[slot] = std::move(batch);
            bq.valid[slot].store(true, std::memory_order_release);
        }
    }
}

// =============================================================================
// DISPATCH THREAD
// =============================================================================

void BertOptimizedSUT::dispatch_thread_func(int bucket_idx) {
    while (dispatch_running_.load(std::memory_order_acquire)) {
        auto& bq = batch_queues_[bucket_idx];
        size_t tail = bq.tail.load(std::memory_order_relaxed);
        size_t slot = tail % BATCH_QUEUE_SIZE;

        if (!bq.valid[slot].load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        bq.tail.store(tail + 1, std::memory_order_relaxed);

        BertBatch& batch = bq.batches[slot];
        int seq_len = SEQ_BUCKETS[bucket_idx];
        int batch_size = bucket_batch_sizes_[bucket_idx];
        BertModelConfig config{batch_size, seq_len};

        // Round-robin die
        size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();

        size_t req_id = acquire_request(die_idx, config);
        auto& model_ctx = die_contexts_[die_idx]->models[config];
        BertInferContext* ctx = model_ctx->requests[req_id].get();

        int actual = static_cast<int>(batch.query_ids.size());
        for (int i = 0; i < actual; ++i) {
            ctx->query_ids[i] = batch.query_ids[i];
            ctx->sample_indices[i] = batch.sample_indices[i];
        }
        ctx->actual_batch_size = batch_size;
        ctx->num_dummies = batch.num_dummies;

        int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* type = ctx->token_type_ids_tensor.data<int64_t>();

        for (int i = 0; i < actual; ++i) {
            copy_sample_to_tensor(batch.sample_indices[i], seq_len, ids, mask, type, i);
        }
        for (int i = actual; i < batch_size; ++i) {
            copy_sample_to_tensor(batch.sample_indices[0], seq_len, ids, mask, type, i);
        }

        bq.valid[slot].store(false, std::memory_order_release);

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(actual, std::memory_order_relaxed);
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

std::unordered_map<int, BertPrediction> BertOptimizedSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void BertOptimizedSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

} // namespace mlperf_ov
