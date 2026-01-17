/**
 * pybind11 bindings for C++ SUT.
 *
 * This exposes the high-performance C++ SUT to Python while
 * ensuring the critical inference callbacks run without GIL.
 *
 * SUT types:
 * - ResNetCppSUT: async inference for ResNet50 (single float32 input)
 * - BertCppSUT: async inference for BERT (3x int64 inputs, 2x float32 outputs)
 * - RetinaNetCppSUT: async inference for RetinaNet (1x float32 input, 3x outputs)
 * - WhisperCppSUT: encoder-decoder inference for Whisper (mel spectrogram input, token output)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "resnet_sut_cpp.hpp"
#include "bert_sut_cpp.hpp"
#include "retinanet_sut_cpp.hpp"
#include "whisper_sut_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_sut, m) {
    m.doc() = "C++ SUT for maximum throughput (bypasses Python GIL)";

    py::class_<mlperf_ov::ResNetCppSUT>(m, "ResNetCppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             "Create C++ SUT instance")

        .def("load", &mlperf_ov::ResNetCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),  // Release GIL during load
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::ResNetCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_optimal_nireq", &mlperf_ov::ResNetCppSUT::get_optimal_nireq,
             "Get optimal number of inference requests")

        .def("get_input_name", &mlperf_ov::ResNetCppSUT::get_input_name,
             "Get input tensor name")

        .def("get_output_name", &mlperf_ov::ResNetCppSUT::get_output_name,
             "Get output tensor name")

        .def("start_async",
             [](mlperf_ov::ResNetCppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> input,
                uint64_t query_id,
                int sample_idx) {
                 // Get pointer to data - this is safe because we're passing to start_async
                 // which copies the data to the inference request
                 py::buffer_info buf = input.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size * sizeof(float);

                 // Release GIL during async inference start
                 py::gil_scoped_release release;
                 self.start_async(data, size, query_id, sample_idx);
             },
             py::arg("input"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Start async inference (GIL released)")

        .def("wait_all", &mlperf_ov::ResNetCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),  // Release GIL during wait
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::ResNetCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::ResNetCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::ResNetCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::ResNetCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions", &mlperf_ov::ResNetCppSUT::get_predictions,
             "Get stored predictions")

        .def("set_response_callback",
             [](mlperf_ov::ResNetCppSUT& self, py::function callback) {
                 // Wrap Python callback to be called from C++
                 // Note: This callback will be called from C++ thread,
                 // so we need to acquire GIL
                 self.set_response_callback(
                     [callback](uint64_t query_id, const float* data, size_t size) {
                         // Acquire GIL to call Python
                         py::gil_scoped_acquire acquire;

                         if (data != nullptr && size > 0) {
                             // Create numpy array with COPY of data (not view!)
                             // This is critical - the original data may be reused
                             // by another inference after we return from callback
                             size_t num_elements = size / sizeof(float);
                             py::array_t<float> arr(num_elements);
                             std::memcpy(arr.mutable_data(), data, size);
                             callback(query_id, arr);
                         } else {
                             // Error case - pass None
                             callback(query_id, py::none());
                         }
                     });
             },
             py::arg("callback"),
             "Set response callback (called when inference completes)");

    // BertCppSUT - optimized for BERT with 3 int64 inputs and 2 float outputs
    py::class_<mlperf_ov::BertCppSUT>(m, "BertCppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             "Create BERT C++ SUT instance")

        .def("load", &mlperf_ov::BertCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::BertCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_optimal_nireq", &mlperf_ov::BertCppSUT::get_optimal_nireq,
             "Get optimal number of inference requests")

        .def("get_input_ids_name", &mlperf_ov::BertCppSUT::get_input_ids_name,
             "Get input_ids tensor name")

        .def("get_attention_mask_name", &mlperf_ov::BertCppSUT::get_attention_mask_name,
             "Get attention_mask tensor name")

        .def("get_token_type_ids_name", &mlperf_ov::BertCppSUT::get_token_type_ids_name,
             "Get token_type_ids tensor name")

        .def("get_start_logits_name", &mlperf_ov::BertCppSUT::get_start_logits_name,
             "Get start_logits output name")

        .def("get_end_logits_name", &mlperf_ov::BertCppSUT::get_end_logits_name,
             "Get end_logits output name")

        .def("get_seq_length", &mlperf_ov::BertCppSUT::get_seq_length,
             "Get sequence length")

        .def("start_async",
             [](mlperf_ov::BertCppSUT& self,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> input_ids,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> attention_mask,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> token_type_ids,
                uint64_t query_id,
                int sample_idx) {
                 py::buffer_info ids_buf = input_ids.request();
                 py::buffer_info mask_buf = attention_mask.request();
                 py::buffer_info type_buf = token_type_ids.request();

                 const int64_t* ids_data = static_cast<const int64_t*>(ids_buf.ptr);
                 const int64_t* mask_data = static_cast<const int64_t*>(mask_buf.ptr);
                 const int64_t* type_data = static_cast<const int64_t*>(type_buf.ptr);
                 int seq_length = static_cast<int>(ids_buf.size);

                 // Release GIL during async inference
                 py::gil_scoped_release release;
                 self.start_async(ids_data, mask_data, type_data, seq_length, query_id, sample_idx);
             },
             py::arg("input_ids"),
             py::arg("attention_mask"),
             py::arg("token_type_ids"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Start async BERT inference (GIL released)")

        .def("wait_all", &mlperf_ov::BertCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::BertCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::BertCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::BertCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::BertCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::BertCppSUT& self) {
                 // Convert C++ predictions to Python dict
                 auto cpp_preds = self.get_predictions();
                 py::dict py_preds;

                 for (const auto& [idx, pred] : cpp_preds) {
                     py::dict entry;
                     py::array_t<float> start_arr(pred.start_logits.size());
                     py::array_t<float> end_arr(pred.end_logits.size());

                     std::memcpy(start_arr.mutable_data(), pred.start_logits.data(),
                                pred.start_logits.size() * sizeof(float));
                     std::memcpy(end_arr.mutable_data(), pred.end_logits.data(),
                                pred.end_logits.size() * sizeof(float));

                     entry["start_logits"] = start_arr;
                     entry["end_logits"] = end_arr;
                     py_preds[py::int_(idx)] = entry;
                 }
                 return py_preds;
             },
             "Get stored predictions as dict of {sample_idx: {start_logits, end_logits}}")

        .def("set_response_callback",
             [](mlperf_ov::BertCppSUT& self, py::function callback) {
                 self.set_response_callback(
                     [callback](uint64_t query_id,
                               const float* start_data,
                               const float* end_data,
                               size_t logits_size) {
                         // Acquire GIL to call Python
                         py::gil_scoped_acquire acquire;

                         if (start_data != nullptr && end_data != nullptr && logits_size > 0) {
                             // Copy data to numpy arrays
                             py::array_t<float> start_arr(logits_size);
                             py::array_t<float> end_arr(logits_size);

                             std::memcpy(start_arr.mutable_data(), start_data, logits_size * sizeof(float));
                             std::memcpy(end_arr.mutable_data(), end_data, logits_size * sizeof(float));

                             callback(query_id, start_arr, end_arr);
                         } else {
                             callback(query_id, py::none(), py::none());
                         }
                     });
             },
             py::arg("callback"),
             "Set response callback (receives query_id, start_logits, end_logits)");

    // RetinaNetCppSUT - optimized for RetinaNet Object Detection
    py::class_<mlperf_ov::RetinaNetCppSUT>(m, "RetinaNetCppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             "Create RetinaNet C++ SUT instance")

        .def("load", &mlperf_ov::RetinaNetCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::RetinaNetCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_optimal_nireq", &mlperf_ov::RetinaNetCppSUT::get_optimal_nireq,
             "Get optimal number of inference requests")

        .def("get_input_name", &mlperf_ov::RetinaNetCppSUT::get_input_name,
             "Get input tensor name")

        .def("get_boxes_name", &mlperf_ov::RetinaNetCppSUT::get_boxes_name,
             "Get boxes output name")

        .def("get_scores_name", &mlperf_ov::RetinaNetCppSUT::get_scores_name,
             "Get scores output name")

        .def("get_labels_name", &mlperf_ov::RetinaNetCppSUT::get_labels_name,
             "Get labels output name")

        .def("get_input_shape", &mlperf_ov::RetinaNetCppSUT::get_input_shape,
             "Get input shape")

        .def("start_async",
             [](mlperf_ov::RetinaNetCppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> input,
                uint64_t query_id,
                int sample_idx) {
                 py::buffer_info buf = input.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size;

                 py::gil_scoped_release release;
                 self.start_async(data, size, query_id, sample_idx);
             },
             py::arg("input"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Start async RetinaNet inference (GIL released)")

        .def("wait_all", &mlperf_ov::RetinaNetCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::RetinaNetCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::RetinaNetCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::RetinaNetCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::RetinaNetCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::RetinaNetCppSUT& self) {
                 auto cpp_preds = self.get_predictions();
                 py::dict py_preds;

                 for (const auto& [idx, pred] : cpp_preds) {
                     py::dict entry;

                     py::array_t<float> boxes_arr(pred.boxes.size());
                     py::array_t<float> scores_arr(pred.scores.size());
                     py::array_t<float> labels_arr(pred.labels.size());

                     std::memcpy(boxes_arr.mutable_data(), pred.boxes.data(),
                                pred.boxes.size() * sizeof(float));
                     std::memcpy(scores_arr.mutable_data(), pred.scores.data(),
                                pred.scores.size() * sizeof(float));
                     if (!pred.labels.empty()) {
                         std::memcpy(labels_arr.mutable_data(), pred.labels.data(),
                                    pred.labels.size() * sizeof(float));
                     }

                     entry["boxes"] = boxes_arr;
                     entry["scores"] = scores_arr;
                     entry["labels"] = labels_arr;
                     entry["num_detections"] = pred.num_detections;
                     py_preds[py::int_(idx)] = entry;
                 }
                 return py_preds;
             },
             "Get stored predictions")

        .def("set_response_callback",
             [](mlperf_ov::RetinaNetCppSUT& self, py::function callback) {
                 self.set_response_callback(
                     [callback](uint64_t query_id,
                               const float* boxes_data, size_t boxes_size,
                               const float* scores_data, size_t scores_size,
                               const float* labels_data, size_t labels_size) {
                         py::gil_scoped_acquire acquire;

                         if (boxes_data && scores_data && boxes_size > 0) {
                             py::array_t<float> boxes_arr(boxes_size);
                             py::array_t<float> scores_arr(scores_size);

                             std::memcpy(boxes_arr.mutable_data(), boxes_data, boxes_size * sizeof(float));
                             std::memcpy(scores_arr.mutable_data(), scores_data, scores_size * sizeof(float));

                             if (labels_data && labels_size > 0) {
                                 py::array_t<float> labels_arr(labels_size);
                                 std::memcpy(labels_arr.mutable_data(), labels_data, labels_size * sizeof(float));
                                 callback(query_id, boxes_arr, scores_arr, labels_arr);
                             } else {
                                 callback(query_id, boxes_arr, scores_arr, py::none());
                             }
                         } else {
                             callback(query_id, py::none(), py::none(), py::none());
                         }
                     });
             },
             py::arg("callback"),
             "Set response callback (receives query_id, boxes, scores, labels)");

    // WhisperCppSUT - optimized for Whisper encoder-decoder
    py::class_<mlperf_ov::WhisperCppSUT>(m, "WhisperCppSUT")
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int>(),
             py::arg("encoder_path"),
             py::arg("decoder_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("max_new_tokens") = 440,
             "Create Whisper C++ SUT instance")

        .def("load", &mlperf_ov::WhisperCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile the models")

        .def("is_loaded", &mlperf_ov::WhisperCppSUT::is_loaded,
             "Check if models are loaded")

        .def("get_optimal_nireq", &mlperf_ov::WhisperCppSUT::get_optimal_nireq,
             "Get optimal number of inference requests")

        .def("get_encoder_input_name", &mlperf_ov::WhisperCppSUT::get_encoder_input_name,
             "Get encoder input tensor name")

        .def("get_n_mels", &mlperf_ov::WhisperCppSUT::get_n_mels,
             "Get number of mel bins")

        .def("get_n_frames", &mlperf_ov::WhisperCppSUT::get_n_frames,
             "Get number of frames")

        .def("process_sample",
             [](mlperf_ov::WhisperCppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> mel_features,
                uint64_t query_id,
                int sample_idx) {
                 py::buffer_info buf = mel_features.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size;

                 std::vector<int64_t> tokens;
                 {
                     py::gil_scoped_release release;
                     tokens = self.process_sample(data, size, query_id, sample_idx);
                 }

                 // Convert to numpy array
                 py::array_t<int64_t> result(tokens.size());
                 std::memcpy(result.mutable_data(), tokens.data(), tokens.size() * sizeof(int64_t));
                 return result;
             },
             py::arg("mel_features"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Process a single sample and return generated tokens")

        .def("start_async",
             [](mlperf_ov::WhisperCppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> mel_features,
                uint64_t query_id,
                int sample_idx) {
                 py::buffer_info buf = mel_features.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size;

                 py::gil_scoped_release release;
                 self.start_async(data, size, query_id, sample_idx);
             },
             py::arg("mel_features"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Start async Whisper inference (GIL released)")

        .def("wait_all", &mlperf_ov::WhisperCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::WhisperCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::WhisperCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::WhisperCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::WhisperCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::WhisperCppSUT& self) {
                 auto cpp_preds = self.get_predictions();
                 py::dict py_preds;

                 for (const auto& [idx, pred] : cpp_preds) {
                     py::array_t<int64_t> tokens_arr(pred.tokens.size());
                     std::memcpy(tokens_arr.mutable_data(), pred.tokens.data(),
                                pred.tokens.size() * sizeof(int64_t));
                     py_preds[py::int_(idx)] = tokens_arr;
                 }
                 return py_preds;
             },
             "Get stored predictions as dict of {sample_idx: tokens}")

        .def("set_response_callback",
             [](mlperf_ov::WhisperCppSUT& self, py::function callback) {
                 self.set_response_callback(
                     [callback](uint64_t query_id,
                               const int64_t* tokens,
                               size_t num_tokens) {
                         py::gil_scoped_acquire acquire;

                         if (tokens != nullptr && num_tokens > 0) {
                             py::array_t<int64_t> tokens_arr(num_tokens);
                             std::memcpy(tokens_arr.mutable_data(), tokens,
                                        num_tokens * sizeof(int64_t));
                             callback(query_id, tokens_arr);
                         } else {
                             callback(query_id, py::none());
                         }
                     });
             },
             py::arg("callback"),
             "Set response callback (receives query_id, tokens)");

    // Version info
    m.attr("__version__") = "1.3.0";
}
