/**
 * RetinaNet Multi-Die SUT implementation.
 *
 * Handles RetinaNet object detection with 3 outputs (boxes, scores, labels)
 * on multi-die NPU accelerators.
 */

#include "retinanet_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <chrono>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/properties.hpp>

namespace mlperf_ov {

static constexpr int SLOT_FREE = -1;

RetinaNetMultiDieCppSUT::RetinaNetMultiDieCppSUT(
    const std::string& model_path,
    const std::string& device_prefix,
    int batch_size,
    const std::unordered_map<std::string, std::string>& compile_properties,
    bool use_nhwc_input,
    int nireq_multiplier)
    : model_path_(model_path),
      device_prefix_(device_prefix),
      batch_size_(batch_size),
      compile_properties_(compile_properties),
      use_nhwc_input_(use_nhwc_input),
      nireq_multiplier_(nireq_multiplier) {

    for (int i = 0; i < MAX_REQUESTS; ++i) {
        request_slots_[i].store(SLOT_FREE, std::memory_order_relaxed);
    }
    for (int i = 0; i < WORK_QUEUE_SIZE; ++i) {
        work_queue_[i].valid.store(false, std::memory_order_relaxed);
    }
}

RetinaNetMultiDieCppSUT::~RetinaNetMultiDieCppSUT() {
    issue_running_.store(false, std::memory_order_release);

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
// REQUEST POOL
// =============================================================================

size_t RetinaNetMultiDieCppSUT::acquire_request() {
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

void RetinaNetMultiDieCppSUT::release_request(size_t id) {
    request_slots_[id].store(SLOT_FREE, std::memory_order_release);
}

size_t RetinaNetMultiDieCppSUT::acquire_request_for_die(size_t die_idx) {
    RetinaNetDieContext* die = die_contexts_[die_idx].get();
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
// ISSUE THREAD (per-die)
// =============================================================================

void RetinaNetMultiDieCppSUT::issue_thread_func(size_t die_idx) {
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

        // Try to claim this slot
        if (!work_tail_.compare_exchange_weak(tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }

        idle_spins = 0;
        uint64_t query_id = work_queue_[idx].query_id;
        int sample_idx = work_queue_[idx].sample_idx;
        work_queue_[idx].valid.store(false, std::memory_order_release);

        // Find sample data
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
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Acquire request for this die
        size_t req_id = acquire_request_for_die(die_idx);
        RetinaNetMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;
        ctx->num_dummies = 0;

        // Copy data
        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, sample_data, std::min(sample_size, input_byte_size_));

        // Submit
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
            queued_count_.fetch_sub(1, std::memory_order_relaxed);
            mlperf::QuerySampleResponse response{query_id, 0, 0};
            mlperf::QuerySamplesComplete(&response, 1);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        size_t req_id = acquire_request_for_die(die_idx);
        RetinaNetMultiDieInferContext* ctx = infer_contexts_[req_id].get();

        ctx->query_ids[0] = query_id;
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;
        ctx->num_dummies = 0;

        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, sample_data, std::min(sample_size, input_byte_size_));

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
        queued_count_.fetch_sub(1, std::memory_order_relaxed);
    }
}

// =============================================================================
// INFERENCE COMPLETE CALLBACK
// =============================================================================

void RetinaNetMultiDieCppSUT::on_inference_complete(RetinaNetMultiDieInferContext* ctx) {
    int actual_batch_size = ctx->actual_batch_size;
    int num_dummies = ctx->num_dummies;
    int real_samples = actual_batch_size - num_dummies;

    // Store predictions if needed (only for real samples)
    if (store_predictions_ && real_samples > 0) {
        ov::Tensor boxes_tensor = ctx->request.get_output_tensor(boxes_idx_);
        ov::Tensor scores_tensor = ctx->request.get_output_tensor(scores_idx_);
        ov::Tensor labels_tensor = ctx->request.get_output_tensor(labels_idx_);

        std::lock_guard<std::mutex> lock(predictions_mutex_);

        for (int i = 0; i < real_samples; ++i) {
            int sample_idx = ctx->sample_indices[i];
            RetinaNetDetection det;

            // Extract boxes for this sample
            if (boxes_per_sample_ > 0) {
                const float* boxes_data = boxes_tensor.data<float>() + (i * boxes_per_sample_ * 4);
                det.boxes.assign(boxes_data, boxes_data + boxes_per_sample_ * 4);
            }

            // Extract scores for this sample
            if (scores_per_sample_ > 0) {
                auto scores_type = scores_tensor.get_element_type();
                if (scores_type == ov::element::f32) {
                    const float* scores_data = scores_tensor.data<float>() + (i * scores_per_sample_);
                    det.scores.assign(scores_data, scores_data + scores_per_sample_);
                } else if (scores_type == ov::element::f16) {
                    const ov::float16* f16_data = scores_tensor.data<ov::float16>() + (i * scores_per_sample_);
                    det.scores.resize(scores_per_sample_);
                    for (size_t j = 0; j < scores_per_sample_; ++j) {
                        det.scores[j] = static_cast<float>(f16_data[j]);
                    }
                }
            }

            // Extract labels for this sample
            if (labels_per_sample_ > 0) {
                auto labels_type = labels_tensor.get_element_type();
                if (labels_type == ov::element::f32) {
                    const float* labels_data = labels_tensor.data<float>() + (i * labels_per_sample_);
                    det.labels.assign(labels_data, labels_data + labels_per_sample_);
                } else if (labels_type == ov::element::i64) {
                    const int64_t* i64_data = labels_tensor.data<int64_t>() + (i * labels_per_sample_);
                    det.labels.resize(labels_per_sample_);
                    for (size_t j = 0; j < labels_per_sample_; ++j) {
                        det.labels[j] = static_cast<float>(i64_data[j]);
                    }
                } else if (labels_type == ov::element::i32) {
                    const int32_t* i32_data = labels_tensor.data<int32_t>() + (i * labels_per_sample_);
                    det.labels.resize(labels_per_sample_);
                    for (size_t j = 0; j < labels_per_sample_; ++j) {
                        det.labels[j] = static_cast<float>(i32_data[j]);
                    }
                }
            }

            det.num_detections = static_cast<int>(scores_per_sample_);
            predictions_[sample_idx] = std::move(det);
        }
    }

    // Prepare responses only for real samples (not dummies)
    mlperf::QuerySampleResponse responses[RetinaNetMultiDieInferContext::MAX_BATCH];
    for (int i = 0; i < real_samples; ++i) {
        responses[i] = {ctx->query_ids[i], 0, 0};
    }

    // Release request before calling LoadGen
    size_t pool_id = ctx->pool_id;
    completed_count_.fetch_add(real_samples, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);
    release_request(pool_id);

    // Call LoadGen (only for real samples)
    if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
        if (real_samples > 0) {
            mlperf::QuerySamplesComplete(responses, real_samples);
        }
    } else {
        // Offline mode callback
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

std::vector<std::string> RetinaNetMultiDieCppSUT::discover_dies() {
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

ov::AnyMap RetinaNetMultiDieCppSUT::build_compile_properties() {
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

// =============================================================================
// OUTPUT NAME MAPPING
// =============================================================================

void RetinaNetMultiDieCppSUT::map_output_names() {
    const auto& outputs = model_->outputs();

    // Try to find outputs by name patterns
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string name = outputs[i].get_any_name();
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        if (lower_name.find("box") != std::string::npos || lower_name.find("bbox") != std::string::npos) {
            boxes_name_ = name;
            boxes_idx_ = i;
        } else if (lower_name.find("score") != std::string::npos || lower_name.find("conf") != std::string::npos) {
            scores_name_ = name;
            scores_idx_ = i;
        } else if (lower_name.find("label") != std::string::npos || lower_name.find("class") != std::string::npos) {
            labels_name_ = name;
            labels_idx_ = i;
        }
    }

    // Fallback: assign by index if not found
    if (boxes_name_.empty() && outputs.size() >= 1) {
        boxes_name_ = outputs[0].get_any_name();
        boxes_idx_ = 0;
    }
    if (scores_name_.empty() && outputs.size() >= 2) {
        scores_name_ = outputs[1].get_any_name();
        scores_idx_ = 1;
    }
    if (labels_name_.empty() && outputs.size() >= 3) {
        labels_name_ = outputs[2].get_any_name();
        labels_idx_ = 2;
    }

    // Calculate per-sample output sizes
    auto boxes_shape = model_->outputs()[boxes_idx_].get_partial_shape().get_min_shape();
    auto scores_shape = model_->outputs()[scores_idx_].get_partial_shape().get_min_shape();

    // boxes: [batch, num_detections, 4] or [batch, num_detections * 4]
    boxes_per_sample_ = 1;
    for (size_t i = 1; i < boxes_shape.size(); ++i) {
        boxes_per_sample_ *= boxes_shape[i];
    }
    boxes_per_sample_ /= 4;  // Number of boxes per sample

    // scores: [batch, num_detections]
    scores_per_sample_ = 1;
    for (size_t i = 1; i < scores_shape.size(); ++i) {
        scores_per_sample_ *= scores_shape[i];
    }

    // labels has same size as scores typically
    labels_per_sample_ = scores_per_sample_;
}

// =============================================================================
// LOAD MODEL
// =============================================================================

void RetinaNetMultiDieCppSUT::load() {
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

    // Map output names for RetinaNet
    map_output_names();

    // Reshape for batch size if needed
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

    // Apply NHWC preprocessing if needed
    if (use_nhwc_input_) {
        ov::preprocess::PrePostProcessor ppp(model_);
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model_ = ppp.build();
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
        input_name_ = model_->inputs()[0].get_any_name();
    }

    // Re-map outputs after preprocessing
    map_output_names();

    input_byte_size_ = 1;
    for (auto d : input_shape_) input_byte_size_ *= d;
    input_byte_size_ *= input_type_.size();

    ov::AnyMap properties = build_compile_properties();

    size_t total_requests = 0;
    for (const auto& device_name : active_devices_) {
        auto die_ctx = std::make_unique<RetinaNetDieContext>();
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
            auto ctx = std::make_unique<RetinaNetMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = total_requests;
            ctx->sut = this;

            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            RetinaNetMultiDieInferContext* ctx_ptr = ctx.get();
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

    // Start issue threads
    issue_running_.store(true, std::memory_order_release);
    for (size_t die_idx = 0; die_idx < die_contexts_.size(); ++die_idx) {
        issue_threads_.emplace_back(&RetinaNetMultiDieCppSUT::issue_thread_func, this, die_idx);
    }

    std::cout << "[RetinaNet SUT] Loaded: " << die_contexts_.size() << " dies, "
              << total_requests << " requests" << std::endl;

    loaded_ = true;
}

// =============================================================================
// PUBLIC API
// =============================================================================

std::vector<std::string> RetinaNetMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

int RetinaNetMultiDieCppSUT::get_total_requests() const {
    return static_cast<int>(infer_contexts_.size());
}

void RetinaNetMultiDieCppSUT::start_async_batch(
    const float* input_data,
    size_t input_size,
    const std::vector<uint64_t>& query_ids,
    const std::vector<int>& sample_indices,
    int actual_batch_size) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t id = acquire_request();
    RetinaNetMultiDieInferContext* ctx = infer_contexts_[id].get();

    actual_batch_size = std::min(actual_batch_size, RetinaNetMultiDieInferContext::MAX_BATCH);
    for (int i = 0; i < actual_batch_size; ++i) {
        ctx->query_ids[i] = query_ids[i];
        ctx->sample_indices[i] = sample_indices[i];
    }
    ctx->actual_batch_size = actual_batch_size;
    ctx->num_dummies = 0;

    float* tensor_data = ctx->input_tensor.data<float>();
    std::memcpy(tensor_data, input_data, std::min(input_size, ctx->input_tensor.get_byte_size()));

    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
}

void RetinaNetMultiDieCppSUT::enable_direct_loadgen(bool enable) {
    use_direct_loadgen_.store(enable, std::memory_order_release);
}

void RetinaNetMultiDieCppSUT::wait_all() {
    while (queued_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RetinaNetMultiDieCppSUT::reset_counters() {
    wait_all();
    issued_count_.store(0, std::memory_order_relaxed);
    completed_count_.store(0, std::memory_order_relaxed);
    queued_count_.store(0, std::memory_order_relaxed);
}

std::unordered_map<int, RetinaNetDetection> RetinaNetMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void RetinaNetMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void RetinaNetMultiDieCppSUT::set_batch_response_callback(BatchResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_response_callback_ = callback;
}

void RetinaNetMultiDieCppSUT::register_sample_data(int sample_idx, const float* data, size_t size) {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_[sample_idx] = {data, size};
}

void RetinaNetMultiDieCppSUT::clear_sample_data() {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_.clear();
}

void RetinaNetMultiDieCppSUT::issue_queries_server_fast(
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

void RetinaNetMultiDieCppSUT::run_server_benchmark(
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

    RetinaNetServerQSL server_qsl(total_sample_count, performance_sample_count);
    RetinaNetServerSUT server_sut(this);

    mlperf::TestSettings test_settings;
    test_settings.scenario = mlperf::TestScenario::Server;
    test_settings.mode = is_accuracy_mode ? mlperf::TestMode::AccuracyOnly : mlperf::TestMode::PerformanceOnly;

    if (!mlperf_conf_path.empty()) {
        test_settings.FromConfig(mlperf_conf_path, "retinanet", "Server");
    }
    if (!user_conf_path.empty()) {
        test_settings.FromConfig(user_conf_path, "retinanet", "Server");
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
    std::cout << "[RetinaNet Server] Starting " << mode_str << " run: "
              << "qps=" << test_settings.server_target_qps
              << ", latency=" << test_settings.server_target_latency_ns / 1e6 << "ms"
              << ", samples=" << (is_accuracy_mode ? total_sample_count : performance_sample_count)
              << std::endl;

    mlperf::StartTest(&server_sut, &server_qsl, test_settings, log_settings);

    std::cout << "[RetinaNet Server] Completed: " << completed_count_.load() << " samples" << std::endl;
}

} // namespace mlperf_ov
