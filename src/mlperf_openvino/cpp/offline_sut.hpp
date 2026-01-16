/**
 * C++ Offline SUT for maximum throughput with batch inference.
 *
 * Optimized for Offline scenario where all samples are known upfront:
 * - Sync batch inference (multiple samples per inference call)
 * - No per-sample callback overhead
 * - Maximum GPU/CPU utilization with large batches
 */

#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace mlperf_ov {

/**
 * High-performance Offline SUT using batch inference.
 *
 * Key differences from Server SUT:
 * - Processes multiple samples in one inference call
 * - No async callbacks - sync inference for simpler, faster execution
 * - Optimized for throughput, not latency
 */
class CppOfflineSUT {
public:
    /**
     * Constructor.
     *
     * @param model_path Path to ONNX or OpenVINO IR model
     * @param device Target device (CPU, GPU, etc.)
     * @param batch_size Batch size for inference
     * @param num_streams Number of inference streams (0 = auto)
     */
    CppOfflineSUT(const std::string& model_path,
                  const std::string& device = "CPU",
                  int batch_size = 32,
                  int num_streams = 0);

    ~CppOfflineSUT();

    /**
     * Load and compile the model.
     */
    void load();

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get batch size.
     */
    int get_batch_size() const { return batch_size_; }

    /**
     * Get input tensor name.
     */
    std::string get_input_name() const;

    /**
     * Get output tensor name.
     */
    std::string get_output_name() const;

    /**
     * Get single sample input size in floats.
     */
    size_t get_sample_size() const { return sample_size_; }

    /**
     * Infer a batch of samples synchronously.
     *
     * @param input_data Pointer to batch input data (batch_size * sample_size floats)
     * @param num_samples Number of samples in this batch (<= batch_size)
     * @return Vector of output data for each sample
     */
    std::vector<std::vector<float>> infer_batch(const float* input_data, int num_samples);

    /**
     * Get number of completed samples.
     */
    uint64_t get_completed_count() const { return completed_count_.load(); }

    /**
     * Reset counters.
     */
    void reset_counters();

private:
    // Model configuration
    std::string model_path_;
    std::string device_;
    int batch_size_;
    int num_streams_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    // Pre-allocated tensors
    ov::Tensor input_tensor_;

    // Model info
    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::Shape single_sample_shape_;
    ov::element::Type input_type_;
    size_t sample_size_;  // Size of single sample in elements
    size_t output_idx_;

    // State
    bool loaded_;
    std::atomic<uint64_t> completed_count_;
    std::mutex infer_mutex_;  // Protect inference
};

} // namespace mlperf_ov
