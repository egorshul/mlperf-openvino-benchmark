/**
 * C++ SUT implementation for 3D UNET on multi-die accelerators.
 */

#include "unet3d_multi_die_sut_cpp.hpp"

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

UNet3DMultiDieCppSUT::UNet3DMultiDieCppSUT(const std::string& model_path,
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
}

UNet3DMultiDieCppSUT::~UNet3DMultiDieCppSUT() {
    wait_all();

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        batch_response_callback_ = nullptr;
    }
}

size_t UNet3DMultiDieCppSUT::acquire_request() {
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

void UNet3DMultiDieCppSUT::release_request(size_t id) {
    request_slots_[id].store(SLOT_FREE, std::memory_order_release);
}

size_t UNet3DMultiDieCppSUT::acquire_request_for_die(size_t die_idx) {
    UNet3DDieContext* die = die_contexts_[die_idx].get();
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

void UNet3DMultiDieCppSUT::on_inference_complete(UNet3DInferContext* ctx) {
    int actual_batch_size = ctx->actual_batch_size;
    int num_dummies = ctx->num_dummies;
    int real_samples = actual_batch_size - num_dummies;

    if (store_predictions_ && real_samples > 0) {
        ov::Tensor output_tensor = ctx->request.get_output_tensor(output_idx_);
        auto actual_type = output_tensor.get_element_type();
        size_t total_elems = single_output_size_ * static_cast<size_t>(actual_batch_size);

        std::vector<float> local_output(total_elems);
        bool copy_ok = true;
        if (actual_type == ov::element::f32) {
            const float* src = output_tensor.data<float>();
            std::memcpy(local_output.data(), src, total_elems * sizeof(float));
        } else if (actual_type == ov::element::f16) {
            const ov::float16* src = output_tensor.data<ov::float16>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::bf16) {
            const ov::bfloat16* src = output_tensor.data<ov::bfloat16>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::i32) {
            const int32_t* src = output_tensor.data<int32_t>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::i64) {
            const int64_t* src = output_tensor.data<int64_t>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::i8) {
            const int8_t* src = output_tensor.data<int8_t>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::u8) {
            const uint8_t* src = output_tensor.data<uint8_t>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::i16) {
            const int16_t* src = output_tensor.data<int16_t>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else if (actual_type == ov::element::f64) {
            const double* src = output_tensor.data<double>();
            for (size_t j = 0; j < total_elems; ++j) {
                local_output[j] = static_cast<float>(src[j]);
            }
        } else {
            std::cerr << "[WARN] Unsupported output type: " << actual_type.get_type_name() << std::endl;
            copy_ok = false;
        }

        if (copy_ok) {
            std::lock_guard<std::mutex> lock(predictions_mutex_);
            for (int i = 0; i < real_samples; ++i) {
                int sample_idx = ctx->sample_indices[i];
                const float* sample_output = local_output.data() + (i * single_output_size_);
                predictions_[sample_idx] = std::vector<float>(
                    sample_output, sample_output + single_output_size_);
            }
        }
    }

    mlperf::QuerySampleResponse responses[UNet3DInferContext::MAX_BATCH];
    for (int i = 0; i < real_samples; ++i) {
        responses[i] = {ctx->query_ids[i], 0, 0};
    }

    std::vector<uint64_t> offline_ids;
    if (!use_direct_loadgen_.load(std::memory_order_relaxed) && real_samples > 0) {
        offline_ids.reserve(real_samples);
        for (int i = 0; i < real_samples; ++i) {
            offline_ids.push_back(ctx->query_ids[i]);
        }
    }

    size_t pool_id = ctx->pool_id;
    completed_count_.fetch_add(real_samples, std::memory_order_relaxed);
    pending_count_.fetch_sub(1, std::memory_order_relaxed);

    if (use_direct_loadgen_.load(std::memory_order_relaxed)) {
        if (real_samples > 0) {
            mlperf::QuerySamplesComplete(responses, real_samples);
        }
    } else {
        if (!offline_ids.empty()) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (batch_response_callback_) {
                batch_response_callback_(offline_ids);
            }
        }
    }

    release_request(pool_id);
}

std::vector<std::string> UNet3DMultiDieCppSUT::discover_dies() {
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

    if (dies.empty()) {
        for (const auto& device : all_devices) {
            if (device == device_prefix_) {
                dies.push_back(device);
                break;
            }
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

ov::AnyMap UNet3DMultiDieCppSUT::build_compile_properties() {
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
                unsigned int bs = static_cast<unsigned int>(std::stoul(value));
                properties[ov::optimal_batch_size.name()] = bs;
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

void UNet3DMultiDieCppSUT::load() {
    if (loaded_) return;

    std::cerr << "[3D-UNET] Loading model: " << model_path_ << std::endl;
    std::cerr << "[3D-UNET] Device prefix: " << device_prefix_
              << ", batch_size: " << batch_size_ << std::endl;

    active_devices_ = discover_dies();
    if (active_devices_.empty()) {
        throw std::runtime_error("No " + device_prefix_ + " dies found");
    }

    std::cerr << "[3D-UNET] Found " << active_devices_.size() << " dies: ";
    for (const auto& d : active_devices_) std::cerr << d << " ";
    std::cerr << std::endl;

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

    input_byte_size_ = 1;
    for (auto d : input_shape_) input_byte_size_ *= d;
    input_byte_size_ *= input_type_.size();

    std::cerr << "[3D-UNET] Input shape: ";
    for (auto d : input_shape_) std::cerr << d << " ";
    std::cerr << " (" << input_byte_size_ << " bytes)" << std::endl;

    ov::AnyMap properties = build_compile_properties();

    size_t total_requests = 0;
    for (const auto& device_name : active_devices_) {
        std::cerr << "[3D-UNET] Compiling for " << device_name << "..." << std::endl;
        auto compile_start = std::chrono::steady_clock::now();

        auto die_ctx = std::make_unique<UNet3DDieContext>();
        die_ctx->device_name = device_name;
        die_ctx->compiled_model = core_.compile_model(model_, device_name, properties);

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - compile_start).count();
        std::cerr << "[3D-UNET] Compiled " << device_name << " in " << compile_time << "ms" << std::endl;

        try {
            die_ctx->optimal_nireq = die_ctx->compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (...) {
            die_ctx->optimal_nireq = 4;
        }

        int num_requests = std::max(die_ctx->optimal_nireq * nireq_multiplier_, nireq_multiplier_ * 2);

        die_ctx->request_start_idx = total_requests;

        int actual_requests = 0;
        for (int i = 0; i < num_requests && total_requests < MAX_REQUESTS; ++i) {
            auto ctx = std::make_unique<UNet3DInferContext>();
            ctx->request = die_ctx->compiled_model.create_infer_request();
            ctx->die_name = device_name;
            ctx->pool_id = total_requests;
            ctx->sut = this;

            ctx->input_tensor = ov::Tensor(input_type_, input_shape_);
            ctx->request.set_input_tensor(ctx->input_tensor);

            UNet3DInferContext* ctx_ptr = ctx.get();
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

    auto output_shape = model_->outputs()[output_idx_].get_partial_shape().get_min_shape();
    single_output_size_ = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        single_output_size_ *= output_shape[i];
    }

    std::cerr << "[3D-UNET] Output shape: ";
    for (auto d : output_shape) std::cerr << d << " ";
    std::cerr << " (per-sample size: " << single_output_size_ << " elements)" << std::endl;

    std::cout << "[3D-UNET SUT] Loaded: " << die_contexts_.size() << " dies, "
              << total_requests << " requests" << std::endl;

    loaded_ = true;
}

void UNet3DMultiDieCppSUT::warmup(int iterations) {
    if (!loaded_) return;

    std::cerr << "[3D-UNET] Warming up (" << iterations << " iterations per die)..." << std::endl;

    for (auto& die_ctx : die_contexts_) {
        if (die_ctx->request_count == 0) continue;

        std::cerr << "[3D-UNET] Warmup " << die_ctx->device_name << " " << std::flush;

        size_t req_idx = die_ctx->request_start_idx;
        auto& ctx = infer_contexts_[req_idx];

        float* tensor_data = ctx->input_tensor.data<float>();
        std::memset(tensor_data, 0, ctx->input_tensor.get_byte_size());

        auto warmup_start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ctx->request.infer();
            std::cerr << "." << std::flush;
        }
        auto warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - warmup_start).count();

        std::cerr << " " << warmup_time << "ms"
                  << " (" << (warmup_time / iterations) << "ms avg)" << std::endl;
    }

    std::cerr << "[3D-UNET] Warmup complete" << std::endl;
}

std::vector<std::string> UNet3DMultiDieCppSUT::get_active_devices() const {
    return active_devices_;
}

int UNet3DMultiDieCppSUT::get_total_requests() const {
    return static_cast<int>(infer_contexts_.size());
}

void UNet3DMultiDieCppSUT::start_async_batch(const float* input_data,
                                               size_t input_size,
                                               const std::vector<uint64_t>& query_ids,
                                               const std::vector<int>& sample_indices,
                                               int actual_batch_size) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    size_t id = acquire_request();
    UNet3DInferContext* ctx = infer_contexts_[id].get();
    ctx->request.wait();

    actual_batch_size = std::min(actual_batch_size, UNet3DInferContext::MAX_BATCH);
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

void UNet3DMultiDieCppSUT::enable_direct_loadgen(bool enable) {
    use_direct_loadgen_.store(enable, std::memory_order_release);
}

void UNet3DMultiDieCppSUT::enable_explicit_batching(bool, int, int) {}

void UNet3DMultiDieCppSUT::wait_all() {
    while (queued_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    while (pending_count_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void UNet3DMultiDieCppSUT::reset_counters() {
    wait_all();
    issued_count_.store(0, std::memory_order_relaxed);
    completed_count_.store(0, std::memory_order_relaxed);
    queued_count_.store(0, std::memory_order_relaxed);
}

std::unordered_map<int, std::vector<float>> UNet3DMultiDieCppSUT::get_predictions() const {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    return predictions_;
}

void UNet3DMultiDieCppSUT::clear_predictions() {
    std::lock_guard<std::mutex> lock(predictions_mutex_);
    predictions_.clear();
}

void UNet3DMultiDieCppSUT::set_batch_response_callback(BatchResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_response_callback_ = callback;
}

void UNet3DMultiDieCppSUT::register_sample_data(int sample_idx, const float* data, size_t size) {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_[sample_idx] = {data, size};
}

void UNet3DMultiDieCppSUT::clear_sample_data() {
    std::unique_lock<std::shared_mutex> lock(sample_cache_mutex_);
    sample_data_cache_.clear();
}

} // namespace mlperf_ov
