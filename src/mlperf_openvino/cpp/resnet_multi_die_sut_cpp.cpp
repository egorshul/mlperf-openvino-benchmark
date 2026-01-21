/**
 * High-performance C++ SUT implementation for multi-die accelerators.
 *
 * NVIDIA LWIS-style architecture with per-die parallelism:
 * - IssueQuery: instant return, pushes to shared work queue only
 * - Per-die Issue Threads: each die has own thread pulling from queue,
 *   doing memcpy, and submitting to its device - fully parallel!
 * - Completion Thread: batches responses, calls QuerySamplesComplete
 */

#include "resnet_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <chrono>

#include <openvino/core/preprocess/pre_post_process.hpp>

namespace mlperf_ov {

static constexpr int SLOT_FREE = -1;

ResNetMultiDieCppSUT::ResNetMultiDieCppSUT(const std::string& model_path,
                                           const std::string& device_prefix,
                                           int batch_size,
                                           const std::unordered_map<std::string, std::string>& compile_properties,
                                           bool use_nhwc_input)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      batch_size_(batch_size),
      compile_properties_(compile_properties),
      use_nhwc_input_(use_nhwc_input) {

    for (int i = 0; i < MAX_REQUESTS; ++i) {
        request_slots_[i].store(SLOT_FREE, std::memory_order_relaxed);
    }
    for (int i = 0; i < WORK_QUEUE_SIZE; ++i) {
        work_queue_[i].valid.store(false, std::memory_order_relaxed);
    }
    for (int i = 0; i < COMPLETION_QUEUE_SIZE; ++i) {
        completion_queue_[i].ctx = nullptr;
        completion_queue_[i].valid.store(false, std::memory_order_relaxed);
    }
}

ResNetMultiDieCppSUT::~ResNetMultiDieCppSUT() {
    // Stop threads
    issue_running_.store(false, std::memory_order_release);
    completion_running_.store(false, std::memory_order_release);

    // Join all per-die issue threads
    for (auto& thread : issue_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    issue_threads_.clear();

    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }

    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        batch_response_callback_ = nullptr;
    }
}

// ============================================================================
// LOCK-FREE REQUEST POOL (GLOBAL)
// ============================================================================

size_t ResNetMultiDieCppSUT::acquire_request() {
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
            #elif defined(__aarch64__)
            asm volatile("yield");
            #endif
        } else if (spin_count < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spin_count = 0;
        }
    }
}

void ResNetMultiDieCppSUT::release_request(size_t id) {
    request_slots_[id].store(SLOT_FREE, std::memory_order_release);
}

// ============================================================================
// PER-DIE REQUEST POOL (for parallel issue threads)
// ============================================================================

size_t ResNetMultiDieCppSUT::acquire_request_for_die(size_t die_idx) {
    DieContext* die = die_contexts_[die_idx].get();
    size_t start = die->request_start_idx;
    size_t count = die->request_count;
    size_t hint = die->pool_search_hint.load(std::memory_order_relaxed);

    // Fast path: try from hint
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

    // Spin wait on die's pool only
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
            #elif defined(__aarch64__)
            asm volatile("yield");
            #endif
        } else if (spin_count < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spin_count = 0;
        }
    }
}

void ResNetMultiDieCppSUT::release_request_for_die(size_t /*die_idx*/, size_t global_id) {
    request_slots_[global_id].store(SLOT_FREE, std::memory_order_release);
}

// ============================================================================
// PER-DIE ISSUE THREAD (parallel memcpy + device submission)
// ============================================================================

void ResNetMultiDieCppSUT::issue_thread_func(size_t die_idx) {
    int idle_spins = 0;

    while (issue_running_.load(std::memory_order_acquire)) {
        // Try to claim a work item from shared queue (multiple threads compete)
        size_t tail = work_tail_.load(std::memory_order_relaxed);
        size_t idx = tail % WORK_QUEUE_SIZE;

        if (!work_queue_[idx].valid.load(std::memory_order_acquire)) {
            // No work - low-latency backoff (Server mode needs fast wake-up)
            idle_spins++;
            if (idle_spins < 64) {
                // Tight spin with pause - minimal latency
                #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                #elif defined(__aarch64__)
                asm volatile("yield");
                #endif
            } else if (idle_spins < 256) {
                // Brief yield
                std::this_thread::yield();
            } else {
                // Short sleep, then reset (keeps thread responsive)
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                idle_spins = 0;
            }
            continue;
        }

        // Try to claim this slot (atomic CAS on tail)
        if (!work_tail_.compare_exchange_weak(tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            // Another thread claimed it, retry
            continue;
        }

        // We claimed the work item
        idle_spins = 0;
        uint64_t query_id = work_queue_[idx].query_id;
        int sample_idx = work_queue_[idx].sample_idx;
        work_queue_[idx].valid.store(false, std::memory_order_release);

        // Find sample data (thread-safe, shared read lock for parallelism!)
        const float* sample_data = nullptr;
        size_t sample_size = 0;
        {
            std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
            auto it = sample_data_cache_.find(sample_idx);
            if (it != sample_data_cache_.end()) {
                sample_data = it->second.data;
                sample_size = it->second.size;
            }
        }

        if (!sample_data) {
            // Sample not found - send response to LoadGen anyway (don't hang!)
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Acquire inference request FROM THIS DIE'S POOL (parallel across dies!)
        size_t req_id = acquire_request_for_die(die_idx);
        ResNetMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        // Setup request
        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;

        // Copy data to pre-allocated tensor (parallel memcpy across dies!)
        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, sample_data, std::min(sample_size, input_byte_size_));

        // Submit to THIS DIE's device (parallel submission!)
        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
        queued_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    // Drain remaining work on shutdown (this thread's share)
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

        const float* drain_sample_data = nullptr;
        size_t drain_sample_size = 0;
        {
            std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
            auto it = sample_data_cache_.find(sample_idx);
            if (it != sample_data_cache_.end()) {
                drain_sample_data = it->second.data;
                drain_sample_size = it->second.size;
            }
        }

        if (!drain_sample_data) {
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        size_t req_id = acquire_request_for_die(die_idx);
        ResNetMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;

        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, drain_sample_data, std::min(drain_sample_size, input_byte_size_));

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
        queued_count_.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ============================================================================
// COMPLETION QUEUE
// ============================================================================

void ResNetMultiDieCppSUT::enqueue_completion(ResNetMultiDieInferContext* ctx) {
    // Server mode: INLINE completion for minimum latency
    // This avoids dependency on completion_thread being scheduled
    if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
        int actual_batch_size = ctx->actual_batch_size;

        // Store predictions if needed (accuracy mode)
        if (store_predictions_) {
            ov::Tensor output_tensor = ctx->request.get_output_tensor(output_idx_);
            const float* output_data = nullptr;
            std::vector<float> converted;

            if (output_type_ == ov::element::f32) {
                output_data = output_tensor.data<float>();
            } else if (output_type_ == ov::element::f16) {
                const ov::float16* f16_data = output_tensor.data<ov::float16>();
                converted.resize(single_output_size_ * actual_batch_size);
                for (size_t j = 0; j < converted.size(); ++j) {
                    converted[j] = static_cast<float>(f16_data[j]);
                }
                output_data = converted.data();
            }

            if (output_data) {
                std::lock_guard<std::mutex> lock(predictions_mutex_);
                for (int i = 0; i < actual_batch_size; ++i) {
                    int sample_idx = ctx->sample_indices[i];
                    const float* sample_output = output_data + (i * single_output_size_);
                    predictions_[sample_idx] = std::vector<float>(
                        sample_output, sample_output + single_output_size_);
                }
            }
        }

        // Prepare responses
        mlperf::QuerySampleResponse responses[ResNetMultiDieInferContext::MAX_BATCH];
        for (int i = 0; i < actual_batch_size; ++i) {
            responses[i] = {ctx->query_ids[i], 0, 0};
        }

        // CRITICAL FIX: Release request BEFORE QuerySamplesComplete!
        // This makes request available for reuse immediately.
        size_t pool_id = ctx->pool_id;
        completed_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
        pending_count_.fetch_sub(1, std::memory_order_relaxed);
        release_request(pool_id);

        // Now call LoadGen (request already freed for reuse)
        mlperf::QuerySamplesComplete(responses, actual_batch_size);
        return;
    }

    // Offline mode: use completion queue for batching (throughput > latency)
    size_t head = completion_head_.fetch_add(1, std::memory_order_acq_rel);
    size_t idx = head % COMPLETION_QUEUE_SIZE;

    // Wait if slot is still being read (rare, only if queue is full)
    while (completion_queue_[idx].valid.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    completion_queue_[idx].ctx = ctx;
    completion_queue_[idx].valid.store(true, std::memory_order_release);
}

void ResNetMultiDieCppSUT::completion_thread_func() {
    static constexpr int MAX_BATCH_RESPONSES = 512;
    mlperf::QuerySampleResponse batch_responses[MAX_BATCH_RESPONSES];
    uint64_t callback_ids[MAX_BATCH_RESPONSES];

    int idle_spins = 0;

    while (completion_running_.load(std::memory_order_acquire)) {
        size_t response_count = 0;
        size_t callback_count = 0;
        size_t processed = 0;
        size_t tail = completion_tail_.load(std::memory_order_relaxed);
        int total_samples = 0;

        while (processed < MAX_BATCH_RESPONSES) {
            size_t idx = tail % COMPLETION_QUEUE_SIZE;

            if (!completion_queue_[idx].valid.load(std::memory_order_acquire)) {
                break;
            }

            ResNetMultiDieInferContext* ctx = completion_queue_[idx].ctx;
            completion_queue_[idx].valid.store(false, std::memory_order_release);

            int actual_batch_size = ctx->actual_batch_size;

            if (store_predictions_) {
                ov::Tensor output_tensor = ctx->request.get_output_tensor(output_idx_);
                const float* output_data = nullptr;
                std::vector<float> converted;

                if (output_type_ == ov::element::f32) {
                    output_data = output_tensor.data<float>();
                } else if (output_type_ == ov::element::f16) {
                    const ov::float16* f16_data = output_tensor.data<ov::float16>();
                    converted.resize(single_output_size_ * actual_batch_size);
                    for (size_t j = 0; j < converted.size(); ++j) {
                        converted[j] = static_cast<float>(f16_data[j]);
                    }
                    output_data = converted.data();
                }

                if (output_data) {
                    std::lock_guard<std::mutex> lock(predictions_mutex_);
                    for (int i = 0; i < actual_batch_size; ++i) {
                        int sample_idx = ctx->sample_indices[i];
                        const float* sample_output = output_data + (i * single_output_size_);
                        predictions_[sample_idx] = std::vector<float>(
                            sample_output, sample_output + single_output_size_);
                    }
                }
            }

            // Always use direct LoadGen for maximum performance
            for (int i = 0; i < actual_batch_size && response_count < MAX_BATCH_RESPONSES; ++i) {
                batch_responses[response_count++] = {ctx->query_ids[i], 0, 0};
            }
            // Also collect for Python callback if set (backward compatibility)
            if (!use_direct_loadgen_.load(std::memory_order_relaxed)) {
                for (int i = 0; i < actual_batch_size && callback_count < MAX_BATCH_RESPONSES; ++i) {
                    callback_ids[callback_count++] = ctx->query_ids[i];
                }
            }

            total_samples += actual_batch_size;
            release_request(ctx->pool_id);
            tail++;
            processed++;
        }

        if (total_samples > 0) {
            completed_count_.fetch_add(total_samples, std::memory_order_relaxed);
            pending_count_.fetch_sub(processed, std::memory_order_relaxed);
            completion_tail_.store(tail, std::memory_order_release);
            idle_spins = 0;
        }

        if (response_count > 0) {
            mlperf::QuerySamplesComplete(batch_responses, response_count);
        }

        if (callback_count > 0) {
            std::vector<uint64_t> ids(callback_ids, callback_ids + callback_count);
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (batch_response_callback_) {
                batch_response_callback_(ids);
            }
        }

        if (processed == 0) {
            // Fast wake-up for Server mode - match issue_thread timing
            idle_spins++;
            if (idle_spins < 64) {
                // Tight spin with pause - minimal latency
                #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                #elif defined(__aarch64__)
                asm volatile("yield");
                #endif
            } else if (idle_spins < 256) {
                // Brief yield
                std::this_thread::yield();
            } else {
                // Short sleep, reset counter (stays responsive)
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                idle_spins = 0;
            }
        }
    }

    // Drain on shutdown
    size_t tail = completion_tail_.load(std::memory_order_relaxed);
    while (true) {
        size_t idx = tail % COMPLETION_QUEUE_SIZE;
        if (!completion_queue_[idx].valid.load(std::memory_order_acquire)) break;

        ResNetMultiDieInferContext* ctx = completion_queue_[idx].ctx;
        completion_queue_[idx].valid.store(false, std::memory_order_release);

        if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
            mlperf::QuerySamplesComplete(ctx->responses, ctx->actual_batch_size);
        }

        completed_count_.fetch_add(ctx->actual_batch_size, std::memory_order_relaxed);
        pending_count_.fetch_sub(1, std::memory_order_relaxed);
        release_request(ctx->pool_id);
        tail++;
    }
    completion_tail_.store(tail, std::memory_order_release);
}

// ============================================================================
// DIE DISCOVERY
// ============================================================================

std::vector<std::string> ResNetMultiDieCppSUT::discover_dies() {
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

ov::AnyMap ResNetMultiDieCppSUT::build_compile_properties() {
    ov::AnyMap properties;
    for (const auto& [key, value] : compile_properties_) {
        try { properties[key] = std::stoi(value); continue; } catch (...) {}
        std::string upper = value;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == "TRUE") { properties[key] = true; continue; }
        if (upper == "FALSE") { properties[key] = false; continue; }
        properties[key] = value;
    }
    return properties;
}

// ============================================================================
// LOAD MODEL
// ============================================================================

void ResNetMultiDieCppSUT::load() {
    if (loaded_) return;

    active_devices_ = discover_dies();
    if (active_devices_.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " dies found");
    }

    model_ = core_.read_model(model_path_);

    const auto& inputs = model_->inputs();
    const auto& outputs = model_->outputs();

    if (inputs.empty() || outputs.empty()) {
        throw std::runtime_error("Model has no inputs or outputs");
    }

    input_name_ = inputs[0].get_any_name();
    input_shape_ = inputs[0].get_partial_shape().get_min_shape();
    input_type_ = inputs[0].get_element_type();

    output_idx_ = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_type = outputs[i].get_element_type();
        if (out_type == ov::element::f32 || out_type == ov::element::f16) {
            output_idx_ = i;
            break;
        }
    }

    output_name_ = outputs[output_idx_].get_any_name();
    output_type_ = outputs[output_idx_].get_element_type();

    if (batch_size_ > 1 || input_shape_[0] == 0) {
        std::map<std::string, ov::PartialShape> new_shapes;
        for (const auto& input : inputs) {
            ov::PartialShape new_shape = input.get_partial_shape();
            new_shape[0] = batch_size_;
            new_shapes[input.get_any_name()] = new_shape;
        }
        model_->reshape(new_shapes);
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
    }

    if (use_nhwc_input_) {
        ov::preprocess::PrePostProcessor ppp(model_);
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model_ = ppp.build();
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
        input_name_ = model_->inputs()[0].get_any_name();
    }

    // Calculate input byte size
    input_byte_size_ = 1;
    for (auto d : input_shape_) input_byte_size_ *= d;
    input_byte_size_ *= input_type_.size();

    ov::AnyMap properties = build_compile_properties();

    size_t total_requests = 0;
    for (const auto& device_name : active_devices_) {
        auto die_ctx = std::make_unique<DieContext>();
        die_ctx->device_name = device_name;
        die_ctx->compiled_model = core_.compile_model(model_, device_name, properties);

        try {
            die_ctx->optimal_nireq = die_ctx->compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (...) {
            die_ctx->optimal_nireq = 4;
        }

        int num_requests = std::max(die_ctx->optimal_nireq * 8, 32);

        // Track per-die request pool
        die_ctx->request_start_idx = total_requests;

        int actual_requests = 0;
        for (int i = 0; i < num_requests && total_requests < MAX_REQUESTS; ++i) {
            auto ctx = std::make_unique<ResNetMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = total_requests;
            ctx->sut = this;

            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            ResNetMultiDieInferContext* ctx_ptr = ctx.get();
            ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                ctx_ptr->sut->enqueue_completion(ctx_ptr);
            });

            infer_contexts_.push_back(std::move(ctx));
            request_slots_[total_requests].store(SLOT_FREE, std::memory_order_relaxed);
            total_requests++;
            actual_requests++;
        }

        die_ctx->request_count = actual_requests;
        die_contexts_.push_back(std::move(die_ctx));
    }

    auto output_shape = model_->outputs()[output_idx_].get_partial_shape().get_min_shape();
    single_output_size_ = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        single_output_size_ *= output_shape[i];
    }

    // Start per-die issue threads (parallel memcpy + device submission!)
    issue_running_.store(true, std::memory_order_release);
    for (size_t die_idx = 0; die_idx < die_contexts_.size(); ++die_idx) {
        issue_threads_.emplace_back(&ResNetMultiDieCppSUT::issue_thread_func, this, die_idx);
    }

    // Start completion thread
    completion_running_.store(true, std::memory_order_release);
    completion_thread_ = std::thread(&ResNetMultiDieCppSUT::completion_thread_func, this);

    loaded_ = true;
}

// ============================================================================
// PUBLIC API
// ============================================================================

std::vector<std::string> ResNetMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

int ResNetMultiDieCppSUT::get_total_requests() const {
    return static_cast<int>(infer_contexts_.size());
}

void ResNetMultiDieCppSUT::start_async_batch(const float* input_data,
                                              size_t input_size,
                                              const std::vector<uint64_t>& query_ids,
                                              const std::vector<int>& sample_indices,
                                              int actual_batch_size) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t id = acquire_request();
    ResNetMultiDieInferContext* ctx = infer_contexts_[id].get();

    actual_batch_size = std::min(actual_batch_size, ResNetMultiDieInferContext::MAX_BATCH);
    for (int i = 0; i < actual_batch_size; ++i) {
        ctx->query_ids[i] = query_ids[i];
        ctx->sample_indices[i] = sample_indices[i];
    }
    ctx->actual_batch_size = actual_batch_size;

    float* tensor_data = ctx->input_tensor.data<float>();
    std::memcpy(tensor_data, input_data, std::min(input_size, ctx->input_tensor.get_byte_size()));

    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
}

void ResNetMultiDieCppSUT::on_inference_complete(ResNetMultiDieInferContext* ctx) {
    // Passthrough - handled in callback
}

void ResNetMultiDieCppSUT::enable_direct_loadgen(bool enable) {
    use_direct_loadgen_.store(enable, std::memory_order_release);
}

void ResNetMultiDieCppSUT::wait_all() {
    // Wait for work queue to drain
    while (queued_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    // Wait for pending inferences
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Wait for completion thread
    while (completion_head_.load(std::memory_order_acquire) !=
           completion_tail_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void ResNetMultiDieCppSUT::reset_counters() {
    wait_all();
    issued_count_.store(0, std::memory_order_relaxed);
    completed_count_.store(0, std::memory_order_relaxed);
    queued_count_.store(0, std::memory_order_relaxed);
    die_index_.store(0, std::memory_order_relaxed);
}

std::unordered_map<int, std::vector<float>> ResNetMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void ResNetMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void ResNetMultiDieCppSUT::set_batch_response_callback(BatchResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_response_callback_ = callback;
}

void ResNetMultiDieCppSUT::register_sample_data(int sample_idx, const float* data, size_t size) {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_[sample_idx] = {data, size};
}

void ResNetMultiDieCppSUT::clear_sample_data() {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_.clear();
}

// ============================================================================
// SERVER MODE: DIRECT PATH (like official OpenVINO MLPerf)
// ============================================================================

void ResNetMultiDieCppSUT::issue_queries_server_fast(
    const std::vector<uint64_t>& query_ids,
    const std::vector<int>& sample_indices) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t num_samples = query_ids.size();

    // DIRECT PATH: Process inline like official OpenVINO MLPerf
    // No work_queue overhead - get request, set inputs, start async directly
    for (size_t i = 0; i < num_samples; ++i) {
        uint64_t query_id = query_ids[i];
        int sample_idx = sample_indices[i];

        // Find sample data (lock-free read after registration)
        const float* sample_data = nullptr;
        size_t sample_size = 0;
        {
            std::shared_lock<std::shared_mutex> lock(sample_cache_mutex_);
            auto it = sample_data_cache_.find(sample_idx);
            if (it != sample_data_cache_.end()) {
                sample_data = it->second.data;
                sample_size = it->second.size;
            }
        }

        if (!sample_data) {
            // Sample not found - send dummy response
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            continue;
        }

        // Acquire request (round-robin across dies for load balancing)
        size_t req_id = acquire_request();
        ResNetMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        // Setup context
        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;

        // Copy input data
        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, sample_data, std::min(sample_size, input_byte_size_));

        // Start async inference
        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

void ResNetMultiDieCppSUT::run_server_benchmark(
    size_t total_sample_count,
    size_t performance_sample_count,
    const std::string& mlperf_conf_path,
    const std::string& user_conf_path,
    const std::string& log_output_dir) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Create pure C++ SUT and QSL
    ResNetServerSUT server_sut(this);
    ResNetServerQSL server_qsl(total_sample_count, performance_sample_count);

    // Configure test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = mlperf::TestScenario::Server;
    test_settings.mode = mlperf::TestMode::PerformanceOnly;

    // Load config files if provided
    if (!mlperf_conf_path.empty()) {
        test_settings.FromConfig(mlperf_conf_path, "resnet50", "Server");
    }
    if (!user_conf_path.empty()) {
        test_settings.FromConfig(user_conf_path, "resnet50", "Server");
    }

    // Configure log settings
    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = log_output_dir;
    log_settings.log_output.copy_summary_to_stdout = true;

    // Run benchmark - entirely in C++, no Python in hot path!
    mlperf::StartTestWithLogSettings(&server_sut, &server_qsl, test_settings, log_settings);
}

} // namespace mlperf_ov
