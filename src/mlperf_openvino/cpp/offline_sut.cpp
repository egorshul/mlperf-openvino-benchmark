/**
 * C++ Offline SUT implementation with batch inference.
 */

#include "offline_sut.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace mlperf_ov {

CppOfflineSUT::CppOfflineSUT(const std::string& model_path,
                             const std::string& device,
                             int batch_size,
                             int num_streams)
    : model_path_(model_path),
      device_(device),
      batch_size_(batch_size),
      num_streams_(num_streams),
      sample_size_(0),
      output_idx_(0),
      loaded_(false),
      completed_count_(0) {
}

CppOfflineSUT::~CppOfflineSUT() {
}

void CppOfflineSUT::load() {
    if (loaded_) {
        return;
    }


    // Read model
    model_ = core_.read_model(model_path_);

    // Get input/output info
    const auto& inputs = model_->inputs();
    const auto& outputs = model_->outputs();

    if (inputs.empty() || outputs.empty()) {
        throw std::runtime_error("Model has no inputs or outputs");
    }

    input_name_ = inputs[0].get_any_name();
    single_sample_shape_ = inputs[0].get_partial_shape().get_min_shape();
    input_type_ = inputs[0].get_element_type();

    // Cache output index
    output_idx_ = outputs.size() > 1 ? 1 : 0;
    output_name_ = outputs[output_idx_].get_any_name();

    // Calculate single sample size
    sample_size_ = 1;
    for (size_t i = 1; i < single_sample_shape_.size(); ++i) {
        sample_size_ *= single_sample_shape_[i];
    }

    // Handle dynamic batch - set to our batch size
    input_shape_ = single_sample_shape_;
    if (input_shape_.size() > 0) {
        input_shape_[0] = batch_size_;
    }

    // Build compile properties - optimize for THROUGHPUT
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;

    if (num_streams_ > 0) {
        properties[ov::hint::num_requests.name()] = num_streams_;
    }

    // CPU-specific optimizations
    if (device_ == "CPU") {
        properties[ov::hint::enable_cpu_pinning.name()] = true;
    }

    // Compile model
    compiled_model_ = core_.compile_model(model_, device_, properties);

    // Create single inference request
    infer_request_ = compiled_model_.create_infer_request();

    // Pre-allocate input tensor with batch shape
    input_tensor_ = ov::Tensor(input_type_, input_shape_);
    infer_request_.set_input_tensor(input_tensor_);

    loaded_ = true;
}

std::string CppOfflineSUT::get_input_name() const {
    return input_name_;
}

std::string CppOfflineSUT::get_output_name() const {
    return output_name_;
}

std::vector<std::vector<float>> CppOfflineSUT::infer_batch(const float* input_data, int num_samples) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    if (num_samples <= 0 || num_samples > batch_size_) {
        throw std::runtime_error("Invalid num_samples");
    }

    std::lock_guard<std::mutex> lock(infer_mutex_);

    // Copy input data to tensor
    float* tensor_data = input_tensor_.data<float>();
    size_t copy_size = num_samples * sample_size_ * sizeof(float);
    std::memcpy(tensor_data, input_data, copy_size);

    // If batch is not full, we still run with full batch but only use first num_samples results
    // This is more efficient than reshaping

    // Run sync inference
    infer_request_.infer();

    // Get output tensor
    ov::Tensor output_tensor = infer_request_.get_output_tensor(output_idx_);
    const float* output_data = output_tensor.data<float>();
    size_t output_size_per_sample = output_tensor.get_size() / batch_size_;

    // Extract results for each sample
    std::vector<std::vector<float>> results;
    results.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        const float* sample_output = output_data + i * output_size_per_sample;
        results.emplace_back(sample_output, sample_output + output_size_per_sample);
    }

    completed_count_ += num_samples;
    return results;
}

void CppOfflineSUT::reset_counters() {
    completed_count_ = 0;
}

} // namespace mlperf_ov
