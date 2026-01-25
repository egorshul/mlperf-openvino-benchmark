/**
 * BERT SUT implementation for multi-die NPU accelerators.
 */

#include "bert_multi_die_sut.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <regex>

namespace mlperf_ov {

int BertMultiDieSUT::get_bucket_index(int seq_len) {
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        if (seq_len <= SEQ_BUCKETS[i]) return i;
    }
    return NUM_SEQ_BUCKETS - 1;
}

int BertMultiDieSUT::get_bucket_seq_len(int bucket_idx) {
    if (bucket_idx < 0 || bucket_idx >= NUM_SEQ_BUCKETS) {
        return SEQ_BUCKETS[NUM_SEQ_BUCKETS - 1];
    }
    return SEQ_BUCKETS[bucket_idx];
}

BertMultiDieSUT::BertMultiDieSUT(
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

BertMultiDieSUT::~BertMultiDieSUT() {
    wait_all();
}

void BertMultiDieSUT::set_target_devices(const std::vector<std::string>& devices) {
    target_devices_ = devices;
}

std::vector<std::string> BertMultiDieSUT::get_active_devices() const {
    std::vector<std::string> result;
    for (const auto& die : die_contexts_) {
        result.push_back(die->device_name);
    }
    return result;
}

std::vector<std::pair<int, int>> BertMultiDieSUT::get_model_configs() const {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        int batch = server_mode_ ? SERVER_BATCH_SIZE : OFFLINE_BATCH_SIZES[i];
        result.emplace_back(batch, SEQ_BUCKETS[i]);
    }
    return result;
}

std::vector<std::string> BertMultiDieSUT::discover_devices() {
    if (!target_devices_.empty()) return target_devices_;

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

ov::AnyMap BertMultiDieSUT::build_compile_properties() {
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

void BertMultiDieSUT::detect_io_names() {
    auto inputs = base_model_->inputs();
    for (const auto& input : inputs) {
        std::string name = input.get_any_name();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower.find("input_ids") != std::string::npos ||
            lower.find("input-ids") != std::string::npos) {
            input_ids_name_ = name;
        } else if (lower.find("attention") != std::string::npos ||
                   lower.find("mask") != std::string::npos) {
            attention_mask_name_ = name;
        } else if (lower.find("token_type") != std::string::npos ||
                   lower.find("segment") != std::string::npos) {
            token_type_ids_name_ = name;
        }
    }

    if (input_ids_name_.empty() && inputs.size() >= 1)
        input_ids_name_ = inputs[0].get_any_name();
    if (attention_mask_name_.empty() && inputs.size() >= 2)
        attention_mask_name_ = inputs[1].get_any_name();
    if (token_type_ids_name_.empty() && inputs.size() >= 3)
        token_type_ids_name_ = inputs[2].get_any_name();

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

std::shared_ptr<ov::Model> BertMultiDieSUT::reshape_model(int batch_size, int seq_length) {
    auto model = base_model_->clone();
    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& input : model->inputs()) {
        shapes[input.get_any_name()] = ov::PartialShape{batch_size, seq_length};
    }
    model->reshape(shapes);
    return model;
}

void BertMultiDieSUT::load() {
    if (loaded_) return;

    auto devices = discover_devices();
    if (devices.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " devices found");
    }

    base_model_ = core_.read_model(model_path_);
    detect_io_names();

    auto props = build_compile_properties();

    total_slots_ = devices.size() * NUM_SEQ_BUCKETS * nireq_per_bucket_;
    all_slot_states_ = std::make_unique<std::atomic<int>[]>(total_slots_);
    for (size_t i = 0; i < total_slots_; ++i) {
        all_slot_states_[i].store(SLOT_FREE);
    }

    size_t slot_offset = 0;

    for (size_t die_idx = 0; die_idx < devices.size(); ++die_idx) {
        auto die = std::make_unique<BertDieContext>();
        die->device_name = devices[die_idx];
        die->die_idx = die_idx;

        for (int bucket_idx = 0; bucket_idx < NUM_SEQ_BUCKETS; ++bucket_idx) {
            int batch_size = server_mode_ ? SERVER_BATCH_SIZE : OFFLINE_BATCH_SIZES[bucket_idx];
            int seq_length = SEQ_BUCKETS[bucket_idx];

            auto reshaped = reshape_model(batch_size, seq_length);
            auto compiled = core_.compile_model(reshaped, die->device_name, props);

            auto bucket_model = std::make_unique<BertBucketModel>();
            bucket_model->compiled_model = compiled;
            bucket_model->batch_size = batch_size;
            bucket_model->seq_length = seq_length;
            bucket_model->slot_states = &all_slot_states_[slot_offset];
            bucket_model->num_requests = nireq_per_bucket_;

            for (int r = 0; r < nireq_per_bucket_; ++r) {
                auto ctx = std::make_unique<BertInferContext>();
                ctx->request = compiled.create_infer_request();
                ctx->batch_size = batch_size;
                ctx->seq_length = seq_length;
                ctx->bucket_idx = bucket_idx;
                ctx->die_idx = die_idx;
                ctx->pool_id = r;
                ctx->sut = this;

                ov::Shape shape{static_cast<size_t>(batch_size),
                                static_cast<size_t>(seq_length)};
                ctx->input_ids_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->attention_mask_tensor = ov::Tensor(ov::element::i64, shape);
                ctx->token_type_ids_tensor = ov::Tensor(ov::element::i64, shape);

                ctx->request.set_tensor(input_ids_name_, ctx->input_ids_tensor);
                ctx->request.set_tensor(attention_mask_name_, ctx->attention_mask_tensor);
                ctx->request.set_tensor(token_type_ids_name_, ctx->token_type_ids_tensor);

                BertInferContext* raw = ctx.get();
                ctx->request.set_callback([raw](std::exception_ptr) {
                    raw->sut->on_complete(raw);
                });

                bucket_model->requests.push_back(std::move(ctx));
            }

            slot_offset += nireq_per_bucket_;
            die->bucket_models[bucket_idx] = std::move(bucket_model);
        }

        die_contexts_.push_back(std::move(die));
    }

    loaded_ = true;
}

void BertMultiDieSUT::warmup(int iterations) {
    if (!loaded_) return;

    for (auto& die : die_contexts_) {
        for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
            auto& bucket_model = die->bucket_models[b];
            if (bucket_model->requests.empty()) continue;

            auto& ctx = bucket_model->requests[0];
            int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
            int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
            int64_t* types = ctx->token_type_ids_tensor.data<int64_t>();

            size_t size = ctx->batch_size * ctx->seq_length;
            std::memset(ids, 0, size * sizeof(int64_t));
            std::fill(mask, mask + size, 1);
            std::memset(types, 0, size * sizeof(int64_t));

            for (int i = 0; i < iterations; ++i) {
                ctx->request.infer();
            }
        }
    }
}

void BertMultiDieSUT::build_bucket_cache() {
    if (samples_.empty()) return;

    int max_idx = 0;
    for (const auto& [idx, _] : samples_) {
        if (idx > max_idx) max_idx = idx;
    }

    bucket_cache_.resize(max_idx + 1, -1);
    for (const auto& [idx, sample] : samples_) {
        bucket_cache_[idx] = static_cast<int8_t>(sample.bucket_idx);
    }
    bucket_cache_size_ = max_idx + 1;
}

void BertMultiDieSUT::register_sample(int sample_idx,
                                       const int64_t* input_ids,
                                       const int64_t* attention_mask,
                                       const int64_t* token_type_ids,
                                       int seq_len) {
    int bucket_idx = get_bucket_index(seq_len);
    int bucket_len = SEQ_BUCKETS[bucket_idx];

    BertSample sample;
    sample.input_ids.resize(bucket_len);
    sample.attention_mask.resize(bucket_len);
    sample.token_type_ids.resize(bucket_len);
    sample.seq_len = seq_len;
    sample.bucket_idx = bucket_idx;

    int copy_len = std::min(seq_len, bucket_len);
    std::memcpy(sample.input_ids.data(), input_ids, copy_len * sizeof(int64_t));
    std::memcpy(sample.attention_mask.data(), attention_mask, copy_len * sizeof(int64_t));
    std::memcpy(sample.token_type_ids.data(), token_type_ids, copy_len * sizeof(int64_t));

    if (copy_len < bucket_len) {
        std::memset(sample.input_ids.data() + copy_len, 0,
                    (bucket_len - copy_len) * sizeof(int64_t));
        std::memset(sample.attention_mask.data() + copy_len, 0,
                    (bucket_len - copy_len) * sizeof(int64_t));
        std::memset(sample.token_type_ids.data() + copy_len, 0,
                    (bucket_len - copy_len) * sizeof(int64_t));
    }

    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_[sample_idx] = std::move(sample);
}

void BertMultiDieSUT::clear_samples() {
    std::unique_lock<std::shared_mutex> lock(sample_mutex_);
    samples_.clear();
    samples_staged_ = false;
    bucket_cache_.clear();
    bucket_cache_size_ = 0;

    for (int i = 0; i < NUM_SEQ_BUCKETS; ++i) {
        staged_buckets_[i].input_ids.clear();
        staged_buckets_[i].attention_mask.clear();
        staged_buckets_[i].token_type_ids.clear();
        staged_buckets_[i].samples.clear();
        staged_buckets_[i].sample_to_index.clear();
        staged_buckets_[i].staged = false;
    }
}

void BertMultiDieSUT::stage_samples() {
    if (samples_staged_) return;

    std::shared_lock<std::shared_mutex> lock(sample_mutex_);

    std::vector<size_t> counts(NUM_SEQ_BUCKETS, 0);
    for (const auto& [idx, sample] : samples_) {
        counts[sample.bucket_idx]++;
    }

    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        size_t total = counts[b] * SEQ_BUCKETS[b];
        staged_buckets_[b].input_ids.resize(total);
        staged_buckets_[b].attention_mask.resize(total);
        staged_buckets_[b].token_type_ids.resize(total);
        staged_buckets_[b].samples.reserve(counts[b]);
    }

    std::vector<size_t> offsets(NUM_SEQ_BUCKETS, 0);

    for (const auto& [sample_idx, sample] : samples_) {
        int b = sample.bucket_idx;
        int seq_len = SEQ_BUCKETS[b];
        size_t offset = offsets[b];

        std::memcpy(staged_buckets_[b].input_ids.data() + offset,
                    sample.input_ids.data(), seq_len * sizeof(int64_t));
        std::memcpy(staged_buckets_[b].attention_mask.data() + offset,
                    sample.attention_mask.data(), seq_len * sizeof(int64_t));
        std::memcpy(staged_buckets_[b].token_type_ids.data() + offset,
                    sample.token_type_ids.data(), seq_len * sizeof(int64_t));

        size_t staged_idx = staged_buckets_[b].samples.size();
        staged_buckets_[b].samples.push_back({sample_idx, sample.seq_len, offset});
        staged_buckets_[b].sample_to_index[sample_idx] = staged_idx;

        offsets[b] += seq_len;
    }

    for (int b = 0; b < NUM_SEQ_BUCKETS; ++b) {
        staged_buckets_[b].staged = true;
    }

    build_bucket_cache();
    samples_staged_ = true;
}

BertInferContext* BertMultiDieSUT::acquire_request(size_t die_idx, int bucket_idx) {
    auto& die = die_contexts_[die_idx];
    auto& bucket_model = die->bucket_models[bucket_idx];

    size_t n = bucket_model->num_requests;
    size_t hint = bucket_model->pool_hint.load(std::memory_order_relaxed);

    for (size_t attempts = 0; attempts < n * 2; ++attempts) {
        size_t idx = (hint + attempts) % n;
        int expected = SLOT_FREE;
        if (bucket_model->slot_states[idx].compare_exchange_weak(
                expected, static_cast<int>(idx),
                std::memory_order_acquire, std::memory_order_relaxed)) {
            bucket_model->pool_hint.store((idx + 1) % n, std::memory_order_relaxed);
            return bucket_model->requests[idx].get();
        }
    }

    int spin = 0;
    while (true) {
        for (size_t idx = 0; idx < n; ++idx) {
            int expected = SLOT_FREE;
            if (bucket_model->slot_states[idx].compare_exchange_weak(
                    expected, static_cast<int>(idx),
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return bucket_model->requests[idx].get();
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

void BertMultiDieSUT::release_request(BertInferContext* ctx) {
    auto& die = die_contexts_[ctx->die_idx];
    auto& bucket_model = die->bucket_models[ctx->bucket_idx];
    bucket_model->slot_states[ctx->pool_id].store(SLOT_FREE, std::memory_order_release);
}

void BertMultiDieSUT::copy_to_tensor(int sample_idx, int seq_len,
                                      int64_t* ids, int64_t* mask, int64_t* types, int offset) {
    size_t dst = static_cast<size_t>(offset) * seq_len;

    std::shared_lock<std::shared_mutex> lock(sample_mutex_);
    auto it = samples_.find(sample_idx);
    if (it != samples_.end()) {
        const BertSample& s = it->second;
        std::memcpy(ids + dst, s.input_ids.data(), seq_len * sizeof(int64_t));
        std::memcpy(mask + dst, s.attention_mask.data(), seq_len * sizeof(int64_t));
        std::memcpy(types + dst, s.token_type_ids.data(), seq_len * sizeof(int64_t));
    } else {
        std::memset(ids + dst, 0, seq_len * sizeof(int64_t));
        std::memset(mask + dst, 0, seq_len * sizeof(int64_t));
        std::memset(types + dst, 0, seq_len * sizeof(int64_t));
    }
}

void BertMultiDieSUT::copy_staged_to_tensor(int bucket_idx, size_t staged_idx, int seq_len,
                                             int64_t* ids, int64_t* mask, int64_t* types,
                                             int offset) {
    const auto& bucket = staged_buckets_[bucket_idx];
    size_t src = bucket.samples[staged_idx].offset;
    size_t dst = static_cast<size_t>(offset) * seq_len;

    std::memcpy(ids + dst, bucket.input_ids.data() + src, seq_len * sizeof(int64_t));
    std::memcpy(mask + dst, bucket.attention_mask.data() + src, seq_len * sizeof(int64_t));
    std::memcpy(types + dst, bucket.token_type_ids.data() + src, seq_len * sizeof(int64_t));
}

void BertMultiDieSUT::on_complete(BertInferContext* ctx) {
    int actual = ctx->actual_batch_size;
    int dummies = ctx->num_dummies;
    int real = actual - dummies;

    if (store_predictions_ && real > 0) {
        ov::Tensor start_t = ctx->request.get_output_tensor(start_output_idx_);
        ov::Tensor end_t = single_output_ ? start_t : ctx->request.get_output_tensor(end_output_idx_);

        const float* start_data = start_t.data<float>();
        const float* end_data = end_t.data<float>();
        size_t logits_size = start_t.get_size() / actual;

        std::lock_guard<std::mutex> lock(predictions_mutex_);
        for (int i = 0; i < real; ++i) {
            int idx = ctx->sample_indices[i];
            BertPrediction pred;

            if (single_output_ && start_t.get_shape().back() == 2) {
                pred.start_logits.resize(logits_size / 2);
                pred.end_logits.resize(logits_size / 2);
                const float* ptr = start_data + i * logits_size;
                for (size_t j = 0; j < logits_size / 2; ++j) {
                    pred.start_logits[j] = ptr[j * 2];
                    pred.end_logits[j] = ptr[j * 2 + 1];
                }
            } else if (single_output_) {
                size_t half = logits_size / 2;
                pred.start_logits.assign(start_data + i * logits_size,
                                         start_data + i * logits_size + half);
                pred.end_logits.assign(start_data + i * logits_size + half,
                                       start_data + (i + 1) * logits_size);
            } else {
                pred.start_logits.assign(start_data + i * logits_size,
                                         start_data + (i + 1) * logits_size);
                pred.end_logits.assign(end_data + i * logits_size,
                                       end_data + (i + 1) * logits_size);
            }
            predictions_[idx] = std::move(pred);
        }
    }

    if (use_direct_loadgen_.load(std::memory_order_relaxed) && real > 0) {
        mlperf::QuerySampleResponse responses[BertInferContext::MAX_BATCH];
        for (int i = 0; i < real; ++i) {
            responses[i] = {ctx->query_ids[i], 0, 0};
        }
        mlperf::QuerySamplesComplete(responses, real);
    }

    completed_count_.fetch_add(real, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);

    release_request(ctx);
}

void BertMultiDieSUT::submit_batch(int bucket_idx,
                                    const std::vector<uint64_t>& query_ids,
                                    const std::vector<int>& sample_indices) {
    if (!loaded_ || bucket_idx < 0 || bucket_idx >= NUM_SEQ_BUCKETS) return;

    if (!samples_staged_) stage_samples();

    int batch_size = OFFLINE_BATCH_SIZES[bucket_idx];
    int seq_len = SEQ_BUCKETS[bucket_idx];
    const auto& bucket = staged_buckets_[bucket_idx];

    int n = static_cast<int>(query_ids.size());

    for (int i = 0; i < n; i += batch_size) {
        int actual = std::min(batch_size, n - i);
        int dummies = batch_size - actual;

        size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();
        BertInferContext* ctx = acquire_request(die_idx, bucket_idx);

        for (int j = 0; j < actual; ++j) {
            ctx->query_ids[j] = query_ids[i + j];
            ctx->sample_indices[j] = sample_indices[i + j];
        }
        ctx->actual_batch_size = batch_size;
        ctx->num_dummies = dummies;

        int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
        int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
        int64_t* types = ctx->token_type_ids_tensor.data<int64_t>();

        for (int j = 0; j < actual; ++j) {
            int sidx = sample_indices[i + j];
            auto it = bucket.sample_to_index.find(sidx);
            if (it != bucket.sample_to_index.end()) {
                copy_staged_to_tensor(bucket_idx, it->second, seq_len, ids, mask, types, j);
            } else {
                copy_to_tensor(sidx, seq_len, ids, mask, types, j);
            }
        }

        if (dummies > 0 && actual > 0) {
            int first = sample_indices[i];
            auto it = bucket.sample_to_index.find(first);
            for (int j = actual; j < batch_size; ++j) {
                if (it != bucket.sample_to_index.end()) {
                    copy_staged_to_tensor(bucket_idx, it->second, seq_len, ids, mask, types, j);
                }
            }
        }

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(actual, std::memory_order_relaxed);
    }
}

void BertMultiDieSUT::issue_query(uint64_t query_id, int sample_idx) {
    if (!loaded_) return;

    int bucket_idx = NUM_SEQ_BUCKETS - 1;
    if (sample_idx < bucket_cache_size_ && bucket_cache_[sample_idx] >= 0) {
        bucket_idx = bucket_cache_[sample_idx];
    } else {
        std::shared_lock<std::shared_mutex> lock(sample_mutex_);
        auto it = samples_.find(sample_idx);
        if (it != samples_.end()) {
            bucket_idx = it->second.bucket_idx;
        }
    }

    int seq_len = SEQ_BUCKETS[bucket_idx];
    size_t die_idx = die_round_robin_[bucket_idx].fetch_add(1) % die_contexts_.size();
    BertInferContext* ctx = acquire_request(die_idx, bucket_idx);

    ctx->query_ids[0] = query_id;
    ctx->sample_indices[0] = sample_idx;
    ctx->actual_batch_size = 1;
    ctx->num_dummies = 0;

    int64_t* ids = ctx->input_ids_tensor.data<int64_t>();
    int64_t* mask = ctx->attention_mask_tensor.data<int64_t>();
    int64_t* types = ctx->token_type_ids_tensor.data<int64_t>();

    copy_to_tensor(sample_idx, seq_len, ids, mask, types, 0);

    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(1, std::memory_order_relaxed);
}

void BertMultiDieSUT::issue_queries(const std::vector<uint64_t>& query_ids,
                                     const std::vector<int>& sample_indices) {
    for (size_t i = 0; i < query_ids.size(); ++i) {
        issue_query(query_ids[i], sample_indices[i]);
    }
}

void BertMultiDieSUT::wait_all() {
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BertMultiDieSUT::reset_counters() {
    wait_all();
    issued_count_.store(0);
    completed_count_.store(0);
}

std::unordered_map<int, BertPrediction> BertMultiDieSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void BertMultiDieSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

}  // namespace mlperf_ov
