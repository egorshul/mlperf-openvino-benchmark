/**
 * High-performance C++ SUT implementation for multi-die accelerators.
 *
 * Key optimizations:
 * 1. Lock-free request pool - atomic CAS instead of mutex
 * 2. Dedicated completion thread - batches responses before QuerySamplesComplete
 * 3. Pre-allocated buffers - zero allocations in hot path
 * 4. Direct LoadGen C++ calls - no Python/GIL overhead
 */

#include "resnet_multi_die_sut_cpp.hpp"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <chrono>

#include <openvino/core/preprocess/pre_post_process.hpp>

namespace mlperf_ov {

// Constants for lock-free pool
static constexpr int SLOT_FREE = -1;
static constexpr int SLOT_ACQUIRING = -2;

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

    // Initialize all slots as free
    for (int i = 0; i < MAX_REQUESTS; ++i) {
        request_slots_[i].store(SLOT_FREE, std::memory_order_relaxed);
    }

    // Initialize completion queue
    for (int i = 0; i < COMPLETION_QUEUE_SIZE; ++i) {
        completion_queue_[i].ctx = nullptr;
        completion_queue_[i].valid.store(false, std::memory_order_relaxed);
    }
}

ResNetMultiDieCppSUT::~ResNetMultiDieCppSUT() {
    // Stop completion thread
    completion_running_.store(false, std::memory_order_release);
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
// LOCK-FREE REQUEST POOL
// ============================================================================

size_t ResNetMultiDieCppSUT::acquire_request() {
    size_t num_requests = infer_contexts_.size();
    size_t hint = pool_search_hint_.load(std::memory_order_relaxed);

    // Try from hint position first, then wrap around
    for (size_t attempts = 0; attempts < num_requests * 2; ++attempts) {
        size_t idx = (hint + attempts) % num_requests;

        int expected = SLOT_FREE;
        if (request_slots_[idx].compare_exchange_weak(
                expected, static_cast<int>(idx),
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            // Successfully acquired
            pool_search_hint_.store((idx + 1) % num_requests, std::memory_order_relaxed);
            return idx;
        }
    }

    // All slots busy - spin wait for any slot
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
        // Brief pause to reduce contention
        std::this_thread::yield();
    }
}

void ResNetMultiDieCppSUT::release_request(size_t id) {
    request_slots_[id].store(SLOT_FREE, std::memory_order_release);
}

// ============================================================================
// COMPLETION THREAD WITH RESPONSE BATCHING
// ============================================================================

void ResNetMultiDieCppSUT::enqueue_completion(ResNetMultiDieInferContext* ctx) {
    // Get slot in completion queue (MPSC queue - multiple producers)
    size_t head = completion_head_.fetch_add(1, std::memory_order_acq_rel);
    size_t idx = head % COMPLETION_QUEUE_SIZE;

    // Wait if slot is still being read by consumer (should be rare)
    while (completion_queue_[idx].valid.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    completion_queue_[idx].ctx = ctx;
    completion_queue_[idx].valid.store(true, std::memory_order_release);
}

void ResNetMultiDieCppSUT::completion_thread_func() {
    // Pre-allocated batch buffer for LoadGen responses
    static constexpr int MAX_BATCH_RESPONSES = 256;
    std::vector<mlperf::QuerySampleResponse> batch_responses;
    batch_responses.reserve(MAX_BATCH_RESPONSES);

    // For Offline mode callback
    std::vector<uint64_t> callback_ids;
    callback_ids.reserve(MAX_BATCH_RESPONSES);

    while (completion_running_.load(std::memory_order_acquire)) {
        batch_responses.clear();
        callback_ids.clear();

        size_t processed = 0;
        size_t tail = completion_tail_.load(std::memory_order_relaxed);

        // Collect batch of completions
        while (processed < MAX_BATCH_RESPONSES) {
            size_t idx = tail % COMPLETION_QUEUE_SIZE;

            if (!completion_queue_[idx].valid.load(std::memory_order_acquire)) {
                break;  // No more items
            }

            ResNetMultiDieInferContext* ctx = completion_queue_[idx].ctx;
            completion_queue_[idx].valid.store(false, std::memory_order_release);

            // Process this completion
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

            // Collect responses
            if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
                // Direct LoadGen mode - use pre-allocated response buffer
                for (int i = 0; i < actual_batch_size; ++i) {
                    batch_responses.push_back({ctx->query_ids[i], 0, 0});
                }
            } else {
                // Python callback mode
                for (int i = 0; i < actual_batch_size; ++i) {
                    callback_ids.push_back(ctx->query_ids[i]);
                }
            }

            completed_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
            pending_count_.fetch_sub(1, std::memory_order_relaxed);

            // Release request back to pool
            release_request(ctx->pool_id);

            tail++;
            processed++;
        }

        // Update tail
        if (processed > 0) {
            completion_tail_.store(tail, std::memory_order_release);
        }

        // Send batched responses
        if (!batch_responses.empty()) {
            mlperf::QuerySamplesComplete(batch_responses.data(), batch_responses.size());
        }

        if (!callback_ids.empty()) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (batch_response_callback_) {
                batch_response_callback_(callback_ids);
            }
        }

        // If no work, brief pause
        if (processed == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Drain remaining completions on shutdown
    size_t tail = completion_tail_.load(std::memory_order_relaxed);
    while (true) {
        size_t idx = tail % COMPLETION_QUEUE_SIZE;
        if (!completion_queue_[idx].valid.load(std::memory_order_acquire)) {
            break;
        }

        ResNetMultiDieInferContext* ctx = completion_queue_[idx].ctx;
        completion_queue_[idx].valid.store(false, std::memory_order_release);

        int actual_batch_size = ctx->actual_batch_size;

        if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
            mlperf::QuerySamplesComplete(ctx->responses, actual_batch_size);
        }

        completed_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
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
        try {
            properties[key] = std::stoi(value);
            continue;
        } catch (...) {}

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

    // Reshape for batch
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

    // NHWC support
    if (use_nhwc_input_) {
        ov::preprocess::PrePostProcessor ppp(model_);
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model_ = ppp.build();
        input_shape_ = model_->inputs()[0].get_partial_shape().get_min_shape();
        input_name_ = model_->inputs()[0].get_any_name();
    }

    ov::AnyMap properties = build_compile_properties();

    // Compile for each die
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

        // Create requests - use higher multiplier for better pipelining
        int num_requests = std::max(die_ctx->optimal_nireq * 8, 32);

        for (int i = 0; i < num_requests && total_requests < MAX_REQUESTS; ++i) {
            auto ctx = std::make_unique<ResNetMultiDieInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = total_requests;
            ctx->sut = this;

            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            // Set callback - just enqueues to completion queue (minimal work)
            ResNetMultiDieInferContext* ctx_ptr = ctx.get();
            ctx->request.set_callback([ctx_ptr](std::exception_ptr) {
                ctx_ptr->sut->enqueue_completion(ctx_ptr);
            });

            infer_contexts_.push_back(std::move(ctx));
            request_slots_[total_requests].store(SLOT_FREE, std::memory_order_relaxed);
            total_requests++;
        }

        die_contexts_.push_back(std::move(die_ctx));
    }

    // Calculate output size
    auto output_shape = model_->outputs()[output_idx_].get_partial_shape().get_min_shape();
    single_output_size_ = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        single_output_size_ *= output_shape[i];
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

    // Copy batch info to pre-allocated arrays
    actual_batch_size = std::min(actual_batch_size, ResNetMultiDieInferContext::MAX_BATCH);
    for (int i = 0; i < actual_batch_size; ++i) {
        ctx->query_ids[i] = query_ids[i];
        ctx->sample_indices[i] = sample_indices[i];
    }
    ctx->actual_batch_size = actual_batch_size;

    // Copy input
    float* tensor_data = ctx->input_tensor.data<float>();
    std::memcpy(tensor_data, input_data, std::min(input_size, ctx->input_tensor.get_byte_size()));

    pending_count_.fetch_add(1, std::memory_order_relaxed);
    ctx->request.start_async();
    issued_count_.fetch_add(actual_batch_size, std::memory_order_relaxed);
}

void ResNetMultiDieCppSUT::on_inference_complete(ResNetMultiDieInferContext* ctx) {
    // This is now just a passthrough - the real work happens in callback
    // which calls enqueue_completion
}

void ResNetMultiDieCppSUT::enable_direct_loadgen(bool enable) {
    use_direct_loadgen_.store(enable, std::memory_order_release);
}

void ResNetMultiDieCppSUT::wait_all() {
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Also wait for completion thread to drain
    while (completion_head_.load(std::memory_order_acquire) !=
           completion_tail_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void ResNetMultiDieCppSUT::reset_counters() {
    wait_all();
    issued_count_.store(0, std::memory_order_relaxed);
    completed_count_.store(0, std::memory_order_relaxed);
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
    sample_data_cache_[sample_idx] = {data, size};
}

void ResNetMultiDieCppSUT::clear_sample_data() {
    sample_data_cache_.clear();
}

void ResNetMultiDieCppSUT::issue_queries_server_fast(
    const std::vector<uint64_t>& query_ids,
    const std::vector<int>& sample_indices) {

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t num_samples = query_ids.size();

    for (size_t i = 0; i < num_samples; ++i) {
        int sample_idx = sample_indices[i];

        auto it = sample_data_cache_.find(sample_idx);
        if (it == sample_data_cache_.end()) {
            throw std::runtime_error("Sample " + std::to_string(sample_idx) + " not registered");
        }

        const SampleData& sample = it->second;

        size_t id = acquire_request();
        ResNetMultiDieInferContext* ctx = infer_contexts_[id].get();

        // Single sample - use pre-allocated arrays
        ctx->query_ids[0] = query_ids[i];
        ctx->sample_indices[0] = sample_idx;
        ctx->actual_batch_size = 1;

        float* tensor_data = ctx->input_tensor.data<float>();
        std::memcpy(tensor_data, sample.data, std::min(sample.size, ctx->input_tensor.get_byte_size()));

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        ctx->request.start_async();
        issued_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace mlperf_ov
