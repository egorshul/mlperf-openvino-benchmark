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
 * - ResNetMultiDieCppSUT: multi-die accelerator inference with batching
 * - RetinaNetMultiDieCppSUT: multi-die accelerator inference for RetinaNet
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "resnet_sut_cpp.hpp"
#include "bert_sut_cpp.hpp"
#include "retinanet_sut_cpp.hpp"
#include "resnet_multi_die_sut_cpp.hpp"
#include "retinanet_multi_die_sut_cpp.hpp"

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

    // ResNetMultiDieCppSUT - optimized for multi-die accelerators with batching
    py::class_<mlperf_ov::ResNetMultiDieCppSUT>(m, "ResNetMultiDieCppSUT")
        .def(py::init<const std::string&, const std::string&, int,
                      const std::unordered_map<std::string, std::string>&, bool>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("batch_size") = 1,
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("use_nhwc_input") = false,
             "Create multi-die accelerator SUT")

        .def("load", &mlperf_ov::ResNetMultiDieCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load model and compile for all available dies")

        .def("is_loaded", &mlperf_ov::ResNetMultiDieCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_num_dies", &mlperf_ov::ResNetMultiDieCppSUT::get_num_dies,
             "Get number of active dies")

        .def("get_active_devices", &mlperf_ov::ResNetMultiDieCppSUT::get_active_devices,
             "Get list of active device names")

        .def("get_batch_size", &mlperf_ov::ResNetMultiDieCppSUT::get_batch_size,
             "Get batch size")

        .def("get_total_requests", &mlperf_ov::ResNetMultiDieCppSUT::get_total_requests,
             "Get total number of inference requests across all dies")

        .def("get_input_name", &mlperf_ov::ResNetMultiDieCppSUT::get_input_name,
             "Get input tensor name")

        .def("get_output_name", &mlperf_ov::ResNetMultiDieCppSUT::get_output_name,
             "Get output tensor name")

        .def("start_async_batch",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> input,
                const std::vector<uint64_t>& query_ids,
                const std::vector<int>& sample_indices,
                int actual_batch_size) {
                 py::buffer_info buf = input.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size * sizeof(float);

                 py::gil_scoped_release release;
                 self.start_async_batch(data, size, query_ids, sample_indices, actual_batch_size);
             },
             py::arg("input"),
             py::arg("query_ids"),
             py::arg("sample_indices"),
             py::arg("actual_batch_size"),
             "Start async batch inference (GIL released)")

        .def("wait_all", &mlperf_ov::ResNetMultiDieCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::ResNetMultiDieCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::ResNetMultiDieCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::ResNetMultiDieCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::ResNetMultiDieCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions", &mlperf_ov::ResNetMultiDieCppSUT::get_predictions,
             "Get stored predictions")

        .def("clear_predictions", &mlperf_ov::ResNetMultiDieCppSUT::clear_predictions,
             "Clear stored predictions")

        .def("set_response_callback",
             [](mlperf_ov::ResNetMultiDieCppSUT& self, py::function callback) {
                 self.set_response_callback(
                     [callback](uint64_t query_id, const float* data, size_t size) {
                         py::gil_scoped_acquire acquire;

                         if (data != nullptr && size > 0) {
                             size_t num_elements = size / sizeof(float);
                             py::array_t<float> arr(num_elements);
                             std::memcpy(arr.mutable_data(), data, size);
                             callback(query_id, arr);
                         } else {
                             callback(query_id, py::none());
                         }
                     });
             },
             py::arg("callback"),
             "Set response callback (called for each sample in batch)")

        .def("set_batch_response_callback",
             [](mlperf_ov::ResNetMultiDieCppSUT& self, py::function callback) {
                 self.set_batch_response_callback(
                     [callback](const std::vector<uint64_t>& query_ids) {
                         py::gil_scoped_acquire acquire;
                         callback(query_ids);
                     });
             },
             py::arg("callback"),
             "Set batch response callback (more efficient - one call per batch)")

        .def("set_loadgen_complete",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
                py::object lg_module,
                py::object response_class) {
                 // Store LoadGen module and QuerySampleResponse class
                 // This allows us to call QuerySamplesComplete directly from C++
                 // with minimal Python overhead
                 py::function complete_func = lg_module.attr("QuerySamplesComplete");

                 self.set_batch_response_callback(
                     [complete_func, response_class](const std::vector<uint64_t>& query_ids) {
                         py::gil_scoped_acquire acquire;

                         // Create responses list directly in C++
                         py::list responses;
                         responses.attr("__init__")();

                         for (uint64_t qid : query_ids) {
                             // QuerySampleResponse(id, data_ptr, size) - no data for classification
                             py::object resp = response_class(qid, 0, 0);
                             responses.append(resp);
                         }

                         // Single call to LoadGen
                         complete_func(responses);
                     });
             },
             py::arg("lg_module"),
             py::arg("response_class"),
             "Set LoadGen module for direct C++ calls (most efficient)")

        .def("issue_queries_server",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
                py::list input_arrays,
                py::list query_ids,
                py::list sample_indices) {
                 // Pre-allocate vectors with known size (fast with GIL)
                 size_t n = py::len(input_arrays);

                 // Extract all data pointers FIRST while holding GIL
                 // This is necessary because numpy arrays may be garbage collected
                 std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> arrays;
                 arrays.reserve(n);

                 std::vector<const float*> all_input_data(n);
                 std::vector<size_t> input_sizes(n);
                 std::vector<uint64_t> qids(n);
                 std::vector<int> sidxs(n);

                 // Fast extraction loop - minimize Python API calls
                 for (size_t i = 0; i < n; ++i) {
                     arrays.push_back(input_arrays[i].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>());
                     qids[i] = query_ids[i].cast<uint64_t>();
                     sidxs[i] = sample_indices[i].cast<int>();
                 }

                 // Get buffer info (separate loop to avoid cache misses)
                 for (size_t i = 0; i < n; ++i) {
                     py::buffer_info buf = arrays[i].request();
                     all_input_data[i] = static_cast<const float*>(buf.ptr);
                     input_sizes[i] = buf.size * sizeof(float);
                 }

                 // Release GIL and process all queries in C++
                 // Arrays kept alive by 'arrays' vector until scope ends
                 {
                     py::gil_scoped_release release;
                     self.issue_queries_server(all_input_data, input_sizes, qids, sidxs);
                 }
                 // Arrays released here, after C++ processing complete
             },
             py::arg("input_arrays"),
             py::arg("query_ids"),
             py::arg("sample_indices"),
             "Process multiple queries efficiently in C++ (GIL released during dispatch)")

        .def("flush_pending_responses", &mlperf_ov::ResNetMultiDieCppSUT::flush_pending_responses,
             "Flush any pending batched responses to callback")

        .def("register_sample_data",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
                int sample_idx,
                py::array_t<float, py::array::c_style | py::array::forcecast> data) {
                 py::buffer_info buf = data.request();
                 // Note: data must remain valid until clear_sample_data is called!
                 self.register_sample_data(
                     sample_idx,
                     static_cast<const float*>(buf.ptr),
                     buf.size * sizeof(float));
             },
             py::arg("sample_idx"),
             py::arg("data"),
             "Register sample data for fast dispatch (data must remain valid!)")

        .def("clear_sample_data", &mlperf_ov::ResNetMultiDieCppSUT::clear_sample_data,
             "Clear all registered sample data")

        .def("issue_queries_server_fast",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
                py::list query_ids,
                py::list sample_indices) {
                 // Fast conversion - only integers, no arrays
                 size_t n = py::len(query_ids);
                 std::vector<uint64_t> qids(n);
                 std::vector<int> sidxs(n);

                 for (size_t i = 0; i < n; ++i) {
                     qids[i] = query_ids[i].cast<uint64_t>();
                     sidxs[i] = sample_indices[i].cast<int>();
                 }

                 // Release GIL - all data access is in C++
                 py::gil_scoped_release release;
                 self.issue_queries_server_fast(qids, sidxs);
             },
             py::arg("query_ids"),
             py::arg("sample_indices"),
             "Fast Server mode dispatch using pre-registered data (no data passing)")

        .def("enable_batched_responses", &mlperf_ov::ResNetMultiDieCppSUT::enable_batched_responses,
             py::arg("enable"),
             "Enable batched response mode to reduce GIL overhead")

        .def("enable_direct_loadgen", &mlperf_ov::ResNetMultiDieCppSUT::enable_direct_loadgen,
             py::arg("enable"),
             "Enable direct LoadGen C++ mode (requires -DUSE_LOADGEN_CPP)")

        .def_static("is_direct_loadgen_available",
             &mlperf_ov::ResNetMultiDieCppSUT::is_direct_loadgen_available,
             "Check if direct LoadGen mode is available");

    // RetinaNetMultiDieCppSUT - multi-die accelerator for RetinaNet
    py::class_<mlperf_ov::RetinaNetMultiDieCppSUT>(m, "RetinaNetMultiDieCppSUT")
        .def(py::init<const std::string&, const std::string&,
                      const std::unordered_map<std::string, std::string>&, bool>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("use_nhwc_input") = false,
             "Create multi-die accelerator SUT for RetinaNet")

        .def("load", &mlperf_ov::RetinaNetMultiDieCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load model and compile for all available dies")

        .def("is_loaded", &mlperf_ov::RetinaNetMultiDieCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_num_dies", &mlperf_ov::RetinaNetMultiDieCppSUT::get_num_dies,
             "Get number of active dies")

        .def("get_active_devices", &mlperf_ov::RetinaNetMultiDieCppSUT::get_active_devices,
             "Get list of active device names")

        .def("get_total_requests", &mlperf_ov::RetinaNetMultiDieCppSUT::get_total_requests,
             "Get total number of inference requests")

        .def("get_input_name", &mlperf_ov::RetinaNetMultiDieCppSUT::get_input_name,
             "Get input tensor name")

        .def("get_input_shape", &mlperf_ov::RetinaNetMultiDieCppSUT::get_input_shape,
             "Get input tensor shape")

        .def("start_async",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self,
                py::array_t<float> input,
                uint64_t query_id,
                int sample_idx) {
                 py::gil_scoped_release release;
                 auto buf = input.request();
                 self.start_async(
                     static_cast<const float*>(buf.ptr),
                     buf.size,
                     query_id,
                     sample_idx);
             },
             py::arg("input"),
             py::arg("query_id"),
             py::arg("sample_idx"),
             "Start async inference")

        .def("wait_all", &mlperf_ov::RetinaNetMultiDieCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::RetinaNetMultiDieCppSUT::get_completed_count,
             "Get completed inference count")

        .def("get_issued_count", &mlperf_ov::RetinaNetMultiDieCppSUT::get_issued_count,
             "Get issued inference count")

        .def("reset_counters", &mlperf_ov::RetinaNetMultiDieCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::RetinaNetMultiDieCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self) {
                 auto preds = self.get_predictions();
                 py::dict result;
                 for (const auto& [idx, pred] : preds) {
                     py::dict pred_dict;
                     pred_dict["boxes"] = py::array_t<float>(pred.boxes.size(), pred.boxes.data());
                     pred_dict["scores"] = py::array_t<float>(pred.scores.size(), pred.scores.data());
                     pred_dict["labels"] = py::array_t<float>(pred.labels.size(), pred.labels.data());
                     pred_dict["num_detections"] = pred.num_detections;
                     result[py::int_(idx)] = pred_dict;
                 }
                 return result;
             },
             "Get stored predictions")

        .def("clear_predictions", &mlperf_ov::RetinaNetMultiDieCppSUT::clear_predictions,
             "Clear stored predictions")

        .def("set_response_callback",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self, py::function callback) {
                 self.set_response_callback(
                     [callback](uint64_t query_id,
                                const float* boxes, size_t boxes_size,
                                const float* scores, size_t scores_size,
                                const float* labels, size_t labels_size) {
                         py::gil_scoped_acquire acquire;

                         py::array_t<float> boxes_arr(boxes_size);
                         py::array_t<float> scores_arr(scores_size);
                         py::array_t<float> labels_arr(labels_size);

                         std::memcpy(boxes_arr.mutable_data(), boxes, boxes_size * sizeof(float));
                         std::memcpy(scores_arr.mutable_data(), scores, scores_size * sizeof(float));
                         std::memcpy(labels_arr.mutable_data(), labels, labels_size * sizeof(float));

                         callback(query_id, boxes_arr, scores_arr, labels_arr);
                     });
             },
             py::arg("callback"),
             "Set response callback");

    // Version info
    m.attr("__version__") = "1.0.0";
}
