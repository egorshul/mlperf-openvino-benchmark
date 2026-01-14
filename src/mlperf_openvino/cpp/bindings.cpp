/**
 * pybind11 bindings for C++ SUT.
 *
 * This exposes the high-performance C++ SUT to Python while
 * ensuring the critical inference callbacks run without GIL.
 *
 * Two SUT types:
 * - CppSUT (Server): async inference, batch=1, for Server scenario
 * - CppOfflineSUT (Offline): sync batch inference, for Offline scenario
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "sut_cpp.hpp"
#include "offline_sut.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_sut, m) {
    m.doc() = "C++ SUT for maximum throughput (bypasses Python GIL)";

    py::class_<mlperf_ov::CppSUT>(m, "CppSUT")
        .def(py::init<const std::string&, const std::string&, int, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("num_streams") = 0,
             py::arg("performance_hint") = "THROUGHPUT",
             "Create C++ SUT instance")

        .def("load", &mlperf_ov::CppSUT::load,
             py::call_guard<py::gil_scoped_release>(),  // Release GIL during load
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::CppSUT::is_loaded,
             "Check if model is loaded")

        .def("get_optimal_nireq", &mlperf_ov::CppSUT::get_optimal_nireq,
             "Get optimal number of inference requests")

        .def("get_input_name", &mlperf_ov::CppSUT::get_input_name,
             "Get input tensor name")

        .def("get_output_name", &mlperf_ov::CppSUT::get_output_name,
             "Get output tensor name")

        .def("start_async",
             [](mlperf_ov::CppSUT& self,
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

        .def("wait_all", &mlperf_ov::CppSUT::wait_all,
             py::call_guard<py::gil_scoped_release>(),  // Release GIL during wait
             "Wait for all pending inferences")

        .def("get_completed_count", &mlperf_ov::CppSUT::get_completed_count,
             "Get number of completed samples")

        .def("get_issued_count", &mlperf_ov::CppSUT::get_issued_count,
             "Get number of issued samples")

        .def("reset_counters", &mlperf_ov::CppSUT::reset_counters,
             "Reset counters")

        .def("set_store_predictions", &mlperf_ov::CppSUT::set_store_predictions,
             py::arg("store"),
             "Enable/disable storing predictions")

        .def("get_predictions", &mlperf_ov::CppSUT::get_predictions,
             "Get stored predictions")

        .def("set_response_callback",
             [](mlperf_ov::CppSUT& self, py::function callback) {
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

    // CppOfflineSUT - optimized for Offline scenario with batch inference
    py::class_<mlperf_ov::CppOfflineSUT>(m, "CppOfflineSUT")
        .def(py::init<const std::string&, const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("device") = "CPU",
             py::arg("batch_size") = 32,
             py::arg("num_streams") = 0,
             "Create C++ Offline SUT instance with batch inference")

        .def("load", &mlperf_ov::CppOfflineSUT::load,
             py::call_guard<py::gil_scoped_release>(),
             "Load and compile the model")

        .def("is_loaded", &mlperf_ov::CppOfflineSUT::is_loaded,
             "Check if model is loaded")

        .def("get_batch_size", &mlperf_ov::CppOfflineSUT::get_batch_size,
             "Get batch size")

        .def("get_input_name", &mlperf_ov::CppOfflineSUT::get_input_name,
             "Get input tensor name")

        .def("get_output_name", &mlperf_ov::CppOfflineSUT::get_output_name,
             "Get output tensor name")

        .def("get_sample_size", &mlperf_ov::CppOfflineSUT::get_sample_size,
             "Get single sample size in floats")

        .def("infer_batch",
             [](mlperf_ov::CppOfflineSUT& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> input,
                int num_samples) {
                 py::buffer_info buf = input.request();
                 const float* data = static_cast<const float*>(buf.ptr);

                 // Release GIL during inference
                 std::vector<std::vector<float>> results;
                 {
                     py::gil_scoped_release release;
                     results = self.infer_batch(data, num_samples);
                 }

                 // Convert results to list of numpy arrays
                 py::list py_results;
                 for (const auto& result : results) {
                     py::array_t<float> arr(result.size());
                     std::memcpy(arr.mutable_data(), result.data(), result.size() * sizeof(float));
                     py_results.append(arr);
                 }
                 return py_results;
             },
             py::arg("input"),
             py::arg("num_samples"),
             "Infer a batch of samples (GIL released during inference)")

        .def("get_completed_count", &mlperf_ov::CppOfflineSUT::get_completed_count,
             "Get number of completed samples")

        .def("reset_counters", &mlperf_ov::CppOfflineSUT::reset_counters,
             "Reset counters");

    // Version info
    m.attr("__version__") = "1.0.0";
}
