/**
 * pybind11 bindings for C++ SUT.
 *
 * SUT types:
 * - ResNetCppSUT: async inference for ResNet50 (single float32 input)
 * - BertCppSUT: async inference for BERT (3x int64 inputs, 2x float32 outputs)
 * - RetinaNetCppSUT: async inference for RetinaNet (1x float32 input, 3x outputs)
 * - ResNetMultiDieCppSUT: multi-die accelerator inference with batching
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
#include "bert_multi_die_sut.hpp"
#include "retinanet_multi_die_sut_cpp.hpp"
#include "ssd_resnet34_sut_cpp.hpp"
#include "ssd_resnet34_multi_die_sut_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_sut, m) {
    m.doc() = "C++ SUT for maximum throughput (bypasses Python GIL)";

    py::class_<mlperf_ov::ResNetCppSUT>(m, "ResNetCppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&, bool>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             py::arg("use_nhwc_input") = true,  // NHWC is default
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
        .def(py::init<const std::string&, const std::string&, int, const std::string&, bool>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             py::arg("use_nhwc_input") = true,  // NHWC is default
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
                      const std::unordered_map<std::string, std::string>&, bool, int>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("batch_size") = 1,
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("use_nhwc_input") = true,  // NHWC is default
             py::arg("nireq_multiplier") = 4,
             "Create multi-die accelerator SUT. nireq_multiplier controls queue depth "
             "(lower = less latency for Server mode, higher = more throughput for Offline)")

        .def("load", &mlperf_ov::ResNetMultiDieCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load model and compile for all available dies")

        .def("warmup", &mlperf_ov::ResNetMultiDieCppSUT::warmup,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("iterations") = 2,
             "Run warmup inferences on all dies")

        .def("set_target_devices", &mlperf_ov::ResNetMultiDieCppSUT::set_target_devices,
             py::arg("devices"),
             "Set specific target devices (call before load())")

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

        .def("set_batch_response_callback",
             [](mlperf_ov::ResNetMultiDieCppSUT& self, py::function callback) {
                 self.set_batch_response_callback(
                     [callback](const std::vector<uint64_t>& query_ids) {
                         py::gil_scoped_acquire acquire;
                         callback(query_ids);
                     });
             },
             py::arg("callback"),
             "Set batch response callback for Offline mode")

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
             "Fast Server mode dispatch - responses go directly to LoadGen C++ (NO GIL)")

        .def("enable_direct_loadgen", &mlperf_ov::ResNetMultiDieCppSUT::enable_direct_loadgen,
             py::arg("enable"),
             "Enable direct LoadGen C++ mode for Server scenario")

        .def("enable_explicit_batching", &mlperf_ov::ResNetMultiDieCppSUT::enable_explicit_batching,
             py::arg("enable"),
             py::arg("batch_size") = 4,
             py::arg("timeout_us") = 500,
             "Enable Intel-style explicit batching for Server mode. "
             "batch_size: target batch size (default 4). "
             "timeout_us: max time to wait for batch fill (default 500us)")

        .def("is_explicit_batching_enabled", &mlperf_ov::ResNetMultiDieCppSUT::is_explicit_batching_enabled,
             "Check if explicit batching is enabled")

        .def("run_server_benchmark",
             [](mlperf_ov::ResNetMultiDieCppSUT& self,
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
                 py::gil_scoped_release release;
                 self.run_server_benchmark(
                     total_sample_count,
                     performance_sample_count,
                     mlperf_conf_path,
                     user_conf_path,
                     log_output_dir,
                     target_qps,
                     target_latency_ns,
                     min_duration_ms,
                     min_query_count,
                     is_accuracy_mode);
             },
             py::arg("total_sample_count"),
             py::arg("performance_sample_count"),
             py::arg("mlperf_conf_path") = "",
             py::arg("user_conf_path") = "",
             py::arg("log_output_dir") = ".",
             py::arg("target_qps") = 0.0,
             py::arg("target_latency_ns") = 0,
             py::arg("min_duration_ms") = 0,
             py::arg("min_query_count") = 0,
             py::arg("is_accuracy_mode") = false,
             "Run Server benchmark with pure C++ SUT");

    // BertMultiDieSUT - BERT with sequence length buckets for multi-die NPU
    py::class_<mlperf_ov::BertMultiDieSUT>(m, "BertMultiDieSUT")
        .def(py::init<const std::string&, const std::string&,
                      const std::unordered_map<std::string, std::string>&, int>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("nireq_per_config") = 4,
             "Create optimized BERT SUT with sequence length buckets")

        .def("set_target_devices", &mlperf_ov::BertMultiDieSUT::set_target_devices,
             py::arg("devices"),
             "Set specific target devices (call before load())")

        .def("set_server_mode", &mlperf_ov::BertMultiDieSUT::set_server_mode,
             py::arg("server_mode"),
             "Set Server mode (batch=1 direct inference). Must be called before load().")

        .def("load", &mlperf_ov::BertMultiDieSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile models for all bucket configurations")

        .def("warmup", &mlperf_ov::BertMultiDieSUT::warmup,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("iterations") = 2,
             "Warmup all models with dummy inference")

        .def("is_loaded", &mlperf_ov::BertMultiDieSUT::is_loaded,
             "Check if models are loaded")

        .def("get_num_dies", &mlperf_ov::BertMultiDieSUT::get_num_dies,
             "Get number of active dies")

        .def("get_active_devices", &mlperf_ov::BertMultiDieSUT::get_active_devices,
             "Get list of active device names")

        .def("get_num_model_configs", &mlperf_ov::BertMultiDieSUT::get_num_model_configs,
             "Get number of model configurations")

        .def("get_model_configs", &mlperf_ov::BertMultiDieSUT::get_model_configs,
             "Get model configurations as list of (batch_size, seq_len) pairs")

        .def("register_sample",
             [](mlperf_ov::BertMultiDieSUT& self,
                int sample_idx,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> input_ids,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> attention_mask,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> token_type_ids,
                int actual_seq_len) {
                 py::buffer_info ids_buf = input_ids.request();
                 py::buffer_info mask_buf = attention_mask.request();
                 py::buffer_info type_buf = token_type_ids.request();

                 self.register_sample(
                     sample_idx,
                     static_cast<const int64_t*>(ids_buf.ptr),
                     static_cast<const int64_t*>(mask_buf.ptr),
                     static_cast<const int64_t*>(type_buf.ptr),
                     actual_seq_len);
             },
             py::arg("sample_idx"),
             py::arg("input_ids"),
             py::arg("attention_mask"),
             py::arg("token_type_ids"),
             py::arg("actual_seq_len"),
             "Register sample with sequence length info")

        .def("clear_samples", &mlperf_ov::BertMultiDieSUT::clear_samples,
             "Clear all registered samples")

        .def("stage_samples", &mlperf_ov::BertMultiDieSUT::stage_samples,
             "Stage samples for Offline mode batch copy (optional)")

        .def_static("get_bucket_index", &mlperf_ov::BertMultiDieSUT::get_bucket_index,
             py::arg("seq_len"),
             "Get bucket index for a sequence length")

        .def_static("get_bucket_seq_len", &mlperf_ov::BertMultiDieSUT::get_bucket_seq_len,
             py::arg("bucket_idx"),
             "Get sequence length for a bucket index")

        .def("submit_batch",
             [](mlperf_ov::BertMultiDieSUT& self,
                int bucket_idx,
                py::list query_ids,
                py::list sample_indices) {
                 size_t n = py::len(query_ids);
                 std::vector<uint64_t> qids(n);
                 std::vector<int> sidxs(n);

                 for (size_t i = 0; i < n; ++i) {
                     qids[i] = query_ids[i].cast<uint64_t>();
                     sidxs[i] = sample_indices[i].cast<int>();
                 }

                 py::gil_scoped_release release;
                 self.submit_batch(bucket_idx, qids, sidxs);
             },
             py::arg("bucket_idx"),
             py::arg("query_ids"),
             py::arg("sample_indices"),
             "Submit batch for Offline mode (grouped by bucket)")

        .def("issue_queries",
             [](mlperf_ov::BertMultiDieSUT& self,
                py::list query_ids,
                py::list sample_indices) {
                 size_t n = py::len(query_ids);
                 std::vector<uint64_t> qids(n);
                 std::vector<int> sidxs(n);

                 for (size_t i = 0; i < n; ++i) {
                     qids[i] = query_ids[i].cast<uint64_t>();
                     sidxs[i] = sample_indices[i].cast<int>();
                 }

                 py::gil_scoped_release release;
                 self.issue_queries(qids, sidxs);
             },
             py::arg("query_ids"),
             py::arg("sample_indices"),
             "Issue queries for Server mode (auto-batching)")

        .def("wait_all", &mlperf_ov::BertMultiDieSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("reset_counters", &mlperf_ov::BertMultiDieSUT::reset_counters,
             "Reset counters")

        .def("get_completed_count", &mlperf_ov::BertMultiDieSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::BertMultiDieSUT::get_issued_count,
             "Get number of issued samples")

        .def("set_store_predictions", &mlperf_ov::BertMultiDieSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::BertMultiDieSUT& self) {
                 auto cpp_preds = self.get_predictions();
                 py::dict py_preds;

                 for (const auto& [idx, pred] : cpp_preds) {
                     py::tuple entry = py::make_tuple(
                         py::array_t<float>(pred.start_logits.size(), pred.start_logits.data()),
                         py::array_t<float>(pred.end_logits.size(), pred.end_logits.data())
                     );
                     py_preds[py::int_(idx)] = entry;
                 }
                 return py_preds;
             },
             "Get stored predictions as dict of {sample_idx: (start_logits, end_logits)}")

        .def("clear_predictions", &mlperf_ov::BertMultiDieSUT::clear_predictions,
             "Clear stored predictions")

        .def("enable_direct_loadgen", &mlperf_ov::BertMultiDieSUT::enable_direct_loadgen,
             py::arg("enable"),
             "Enable direct LoadGen C++ mode");

    // RetinaNetMultiDieCppSUT - optimized for multi-die accelerators with batching
    // Note: RetinaNet has larger input (800x800x3=7.7MB vs ResNet 224x224x3=0.6MB)
    // so we use smaller defaults: nireq_multiplier=2, explicit_batch=2, timeout=1000us
    py::class_<mlperf_ov::RetinaNetMultiDieCppSUT>(m, "RetinaNetMultiDieCppSUT")
        .def(py::init<const std::string&, const std::string&, int,
                      const std::unordered_map<std::string, std::string>&, bool, int>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("batch_size") = 1,
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("use_nhwc_input") = true,  // NHWC is default
             py::arg("nireq_multiplier") = 2,   // Lower than ResNet (4) due to large input
             "Create multi-die RetinaNet SUT. nireq_multiplier controls queue depth "
             "(lower = less latency for Server mode, higher = more throughput for Offline)")

        .def("load", &mlperf_ov::RetinaNetMultiDieCppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load model and compile for all available dies")

        .def("warmup", &mlperf_ov::RetinaNetMultiDieCppSUT::warmup,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("iterations") = 2,
             "Run warmup inferences on all dies")

        .def("set_target_devices", &mlperf_ov::RetinaNetMultiDieCppSUT::set_target_devices,
             py::arg("devices"),
             "Set specific target devices (call before load())")

        .def("is_loaded", &mlperf_ov::RetinaNetMultiDieCppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_num_dies", &mlperf_ov::RetinaNetMultiDieCppSUT::get_num_dies,
             "Get number of active dies")

        .def("get_active_devices", &mlperf_ov::RetinaNetMultiDieCppSUT::get_active_devices,
             "Get list of active device names")

        .def("get_batch_size", &mlperf_ov::RetinaNetMultiDieCppSUT::get_batch_size,
             "Get batch size")

        .def("get_total_requests", &mlperf_ov::RetinaNetMultiDieCppSUT::get_total_requests,
             "Get total number of inference requests across all dies")

        .def("get_input_name", &mlperf_ov::RetinaNetMultiDieCppSUT::get_input_name,
             "Get input tensor name")

        .def("get_boxes_name", &mlperf_ov::RetinaNetMultiDieCppSUT::get_boxes_name,
             "Get boxes output name")

        .def("get_scores_name", &mlperf_ov::RetinaNetMultiDieCppSUT::get_scores_name,
             "Get scores output name")

        .def("get_labels_name", &mlperf_ov::RetinaNetMultiDieCppSUT::get_labels_name,
             "Get labels output name")

        .def("start_async_batch",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self,
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

        .def("wait_all", &mlperf_ov::RetinaNetMultiDieCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::RetinaNetMultiDieCppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::RetinaNetMultiDieCppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::RetinaNetMultiDieCppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::RetinaNetMultiDieCppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self) {
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

        .def("clear_predictions", &mlperf_ov::RetinaNetMultiDieCppSUT::clear_predictions,
             "Clear stored predictions")

        .def("set_batch_response_callback",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self, py::function callback) {
                 self.set_batch_response_callback(
                     [callback](const std::vector<uint64_t>& query_ids) {
                         py::gil_scoped_acquire acquire;
                         callback(query_ids);
                     });
             },
             py::arg("callback"),
             "Set batch response callback for Offline mode")

        .def("register_sample_data",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self,
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

        .def("clear_sample_data", &mlperf_ov::RetinaNetMultiDieCppSUT::clear_sample_data,
             "Clear all registered sample data")

        .def("issue_queries_server_fast",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self,
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
             "Fast Server mode dispatch - responses go directly to LoadGen C++ (NO GIL)")

        .def("enable_direct_loadgen", &mlperf_ov::RetinaNetMultiDieCppSUT::enable_direct_loadgen,
             py::arg("enable"),
             "Enable direct LoadGen C++ mode for Server scenario")

        .def("enable_explicit_batching", &mlperf_ov::RetinaNetMultiDieCppSUT::enable_explicit_batching,
             py::arg("enable"),
             py::arg("batch_size") = 2,      // Smaller than ResNet (4) due to 7.7MB input
             py::arg("timeout_us") = 1000,   // Longer timeout (1ms) for larger data copy
             "Enable Intel-style explicit batching for Server mode. "
             "batch_size: target batch size (default 2 for RetinaNet, 4 for ResNet). "
             "timeout_us: max time to wait for batch fill (default 1000us for RetinaNet)")

        .def("is_explicit_batching_enabled", &mlperf_ov::RetinaNetMultiDieCppSUT::is_explicit_batching_enabled,
             "Check if explicit batching is enabled")

        .def("run_server_benchmark",
             [](mlperf_ov::RetinaNetMultiDieCppSUT& self,
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
                 py::gil_scoped_release release;
                 self.run_server_benchmark(
                     total_sample_count,
                     performance_sample_count,
                     mlperf_conf_path,
                     user_conf_path,
                     log_output_dir,
                     target_qps,
                     target_latency_ns,
                     min_duration_ms,
                     min_query_count,
                     is_accuracy_mode);
             },
             py::arg("total_sample_count"),
             py::arg("performance_sample_count"),
             py::arg("mlperf_conf_path") = "",
             py::arg("user_conf_path") = "",
             py::arg("log_output_dir") = ".",
             py::arg("target_qps") = 0.0,
             py::arg("target_latency_ns") = 0,
             py::arg("min_duration_ms") = 0,
             py::arg("min_query_count") = 0,
             py::arg("is_accuracy_mode") = false,
             "Run Server benchmark with pure C++ SUT");

    // =========================================================================
    // SSD-ResNet34 Single-Device SUT
    // =========================================================================
    py::class_<mlperf_ov::SSDResNet34CppSUT>(m, "SSDResNet34CppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&, bool>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             py::arg("use_nhwc_input") = true,
             "Create SSD-ResNet34 C++ SUT instance")

        .def("load", &mlperf_ov::SSDResNet34CppSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::SSDResNet34CppSUT::is_loaded)
        .def("get_optimal_nireq", &mlperf_ov::SSDResNet34CppSUT::get_optimal_nireq)
        .def("get_input_name", &mlperf_ov::SSDResNet34CppSUT::get_input_name)
        .def("get_boxes_name", &mlperf_ov::SSDResNet34CppSUT::get_boxes_name)
        .def("get_scores_name", &mlperf_ov::SSDResNet34CppSUT::get_scores_name)
        .def("get_labels_name", &mlperf_ov::SSDResNet34CppSUT::get_labels_name)
        .def("get_input_shape", &mlperf_ov::SSDResNet34CppSUT::get_input_shape)

        .def("start_async",
             [](mlperf_ov::SSDResNet34CppSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> input,
                uint64_t query_id,
                int sample_idx) {
                 py::buffer_info buf = input.request();
                 const float* data = static_cast<const float*>(buf.ptr);
                 size_t size = buf.size;
                 py::gil_scoped_release release;
                 self.start_async(data, size, query_id, sample_idx);
             },
             py::arg("input"), py::arg("query_id"), py::arg("sample_idx"),
             "Start async SSD-ResNet34 inference (GIL released)")

        .def("wait_all", &mlperf_ov::SSDResNet34CppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>())
        .def("get_completed_count", &mlperf_ov::SSDResNet34CppSUT::get_completed_count)
        .def("get_issued_count", &mlperf_ov::SSDResNet34CppSUT::get_issued_count)
        .def("reset_counters", &mlperf_ov::SSDResNet34CppSUT::reset_counters)
        .def("set_store_predictions", &mlperf_ov::SSDResNet34CppSUT::set_store_predictions,
             py::arg("store"))

        .def("get_predictions",
             [](mlperf_ov::SSDResNet34CppSUT& self) {
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
             [](mlperf_ov::SSDResNet34CppSUT& self, py::function callback) {
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

    // =========================================================================
    // SSD-ResNet34 Multi-Die SUT
    // =========================================================================
    py::class_<mlperf_ov::SSDResNet34MultiDieCppSUT>(m, "SSDResNet34MultiDieCppSUT")
        .def(py::init<const std::string&, const std::string&, int,
                      const std::unordered_map<std::string, std::string>&, bool, int>(),
             py::arg("model_path"),
             py::arg("device_prefix"),
             py::arg("batch_size") = 1,
             py::arg("compile_properties") = std::unordered_map<std::string, std::string>{},
             py::arg("use_nhwc_input") = true,
             py::arg("nireq_multiplier") = 2,
             "Create multi-die SSD-ResNet34 SUT")

        .def("load", &mlperf_ov::SSDResNet34MultiDieCppSUT::load,
             py::call_guard<py::gil_scoped_release>())
        .def("warmup", &mlperf_ov::SSDResNet34MultiDieCppSUT::warmup,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("iterations") = 2)
        .def("set_target_devices", &mlperf_ov::SSDResNet34MultiDieCppSUT::set_target_devices,
             py::arg("devices"))
        .def("is_loaded", &mlperf_ov::SSDResNet34MultiDieCppSUT::is_loaded)
        .def("get_num_dies", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_num_dies)
        .def("get_active_devices", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_active_devices)
        .def("get_batch_size", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_batch_size)
        .def("get_total_requests", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_total_requests)
        .def("get_input_name", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_input_name)
        .def("get_boxes_name", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_boxes_name)
        .def("get_scores_name", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_scores_name)
        .def("get_labels_name", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_labels_name)

        .def("start_async_batch",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self,
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
             py::arg("input"), py::arg("query_ids"),
             py::arg("sample_indices"), py::arg("actual_batch_size"),
             "Start async batch inference (GIL released)")

        .def("wait_all", &mlperf_ov::SSDResNet34MultiDieCppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>())
        .def("get_completed_count", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_completed_count)
        .def("get_issued_count", &mlperf_ov::SSDResNet34MultiDieCppSUT::get_issued_count)
        .def("reset_counters", &mlperf_ov::SSDResNet34MultiDieCppSUT::reset_counters)
        .def("set_store_predictions", &mlperf_ov::SSDResNet34MultiDieCppSUT::set_store_predictions,
             py::arg("store"))

        .def("get_predictions",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self) {
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

        .def("clear_predictions", &mlperf_ov::SSDResNet34MultiDieCppSUT::clear_predictions)

        .def("set_batch_response_callback",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self, py::function callback) {
                 self.set_batch_response_callback(
                     [callback](const std::vector<uint64_t>& query_ids) {
                         py::gil_scoped_acquire acquire;
                         callback(query_ids);
                     });
             },
             py::arg("callback"))

        .def("register_sample_data",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self,
                int sample_idx,
                py::array_t<float, py::array::c_style | py::array::forcecast> data) {
                 py::buffer_info buf = data.request();
                 self.register_sample_data(
                     sample_idx,
                     static_cast<const float*>(buf.ptr),
                     buf.size * sizeof(float));
             },
             py::arg("sample_idx"), py::arg("data"))

        .def("clear_sample_data", &mlperf_ov::SSDResNet34MultiDieCppSUT::clear_sample_data)

        .def("issue_queries_server_fast",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self,
                py::list query_ids,
                py::list sample_indices) {
                 size_t n = py::len(query_ids);
                 std::vector<uint64_t> qids(n);
                 std::vector<int> sidxs(n);
                 for (size_t i = 0; i < n; ++i) {
                     qids[i] = query_ids[i].cast<uint64_t>();
                     sidxs[i] = sample_indices[i].cast<int>();
                 }
                 py::gil_scoped_release release;
                 self.issue_queries_server_fast(qids, sidxs);
             },
             py::arg("query_ids"), py::arg("sample_indices"),
             "Fast Server mode dispatch (NO GIL)")

        .def("enable_direct_loadgen", &mlperf_ov::SSDResNet34MultiDieCppSUT::enable_direct_loadgen,
             py::arg("enable"))

        .def("enable_explicit_batching", &mlperf_ov::SSDResNet34MultiDieCppSUT::enable_explicit_batching,
             py::arg("enable"),
             py::arg("batch_size") = 1,
             py::arg("timeout_us") = 1500,
             "Enable explicit batching for Server mode")

        .def("is_explicit_batching_enabled", &mlperf_ov::SSDResNet34MultiDieCppSUT::is_explicit_batching_enabled)

        .def("run_server_benchmark",
             [](mlperf_ov::SSDResNet34MultiDieCppSUT& self,
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
                 py::gil_scoped_release release;
                 self.run_server_benchmark(
                     total_sample_count, performance_sample_count,
                     mlperf_conf_path, user_conf_path, log_output_dir,
                     target_qps, target_latency_ns, min_duration_ms,
                     min_query_count, is_accuracy_mode);
             },
             py::arg("total_sample_count"),
             py::arg("performance_sample_count"),
             py::arg("mlperf_conf_path") = "",
             py::arg("user_conf_path") = "",
             py::arg("log_output_dir") = ".",
             py::arg("target_qps") = 0.0,
             py::arg("target_latency_ns") = 0,
             py::arg("min_duration_ms") = 0,
             py::arg("min_query_count") = 0,
             py::arg("is_accuracy_mode") = false,
             "Run Server benchmark with pure C++ SUT");

    m.attr("__version__") = "1.0.0";
}
