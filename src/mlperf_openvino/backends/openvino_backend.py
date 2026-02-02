"""
OpenVINO backend implementation for MLPerf Benchmark.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import openvino as ov
    from openvino import Core, CompiledModel, InferRequest, AsyncInferQueue
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    Core = None
    CompiledModel = None
    InferRequest = None
    AsyncInferQueue = None

from .base import BaseBackend
from ..core.config import OpenVINOConfig

logger = logging.getLogger(__name__)


class OpenVINOBackend(BaseBackend):
    """
    OpenVINO backend for inference.

    This backend supports:
    - ONNX models (converted on-the-fly)
    - OpenVINO IR models (.xml/.bin)
    - Various precision modes (FP32, FP16, INT8)
    - Automatic batching and streaming
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[OpenVINOConfig] = None,
        use_nhwc_input: bool = False,
        **kwargs
    ):
        """Initialize OpenVINO backend."""
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO is not installed. Please install it with: "
                "pip install openvino"
            )

        super().__init__(model_path, **kwargs)

        self.config = config or OpenVINOConfig()
        self.use_nhwc_input = use_nhwc_input
        self._core: Optional[Core] = None
        self._model: Optional[ov.Model] = None
        self._compiled_model: Optional[CompiledModel] = None
        self._infer_request: Optional[InferRequest] = None

        # For async inference
        self._infer_requests: List[InferRequest] = []
        self._optimal_nireq: int = 1

        # For thread-safe inference
        self._request_queue: Optional[queue.Queue] = None

        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}

    def load(self) -> None:
        """Load and compile the model."""
        if self._loaded:
            logger.warning("Model already loaded, skipping...")
            return

        logger.info(f"Loading model from {self.model_path}")

        # Validate device - bare accelerator prefix without die number requires MultiDeviceBackend
        if self.config.is_multi_device():
            device_prefix = self.config.get_device_prefix()
            raise ValueError(
                f"Device '{device_prefix}' (all dies) requires MultiDeviceBackend. "
                f"Use a specific die (e.g., '{device_prefix}.0') for OpenVINOBackend, "
                f"or use '--device {device_prefix}' which automatically uses MultiDeviceBackend."
            )

        self._core = Core()

        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._core.set_property({"CACHE_DIR": str(cache_path)})

        model_path = Path(self.model_path)

        if model_path.suffix.lower() in (".onnx", ".xml"):
            self._model = self._core.read_model(str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        self._reshape_model_for_batch(max(self.config.batch_size, 1))

        if self.use_nhwc_input:
            self._apply_nhwc_input_layout()

        self._extract_model_info()

        logger.debug(f"Compiling model for device: {self.config.device}")
        properties = self._build_compile_properties()

        self._compiled_model = self._core.compile_model(
            self._model,
            self.config.device,
            properties
        )

        try:
            self._optimal_nireq = self._compiled_model.get_property(
                ov.properties.optimal_number_of_infer_requests()
            )
        except Exception:
            self._optimal_nireq = 4

        self._create_infer_requests()

        self._loaded = True

    def _build_compile_properties(self) -> Dict[str, Any]:
        """Build compilation properties from config."""
        properties = {}

        if self.config.is_accelerator_device():
            return self._build_accelerator_compile_properties()
        else:
            return self._build_cpu_compile_properties()

    def _build_cpu_compile_properties(self) -> Dict[str, Any]:
        """Build CPU-specific compilation properties."""
        properties = {}

        if self.config.performance_hint:
            hint_enum = getattr(ov.properties.hint.PerformanceMode,
                              self.config.performance_hint, None)
            if hint_enum:
                properties[ov.properties.hint.performance_mode()] = hint_enum

        if self.config.num_streams != "AUTO":
            try:
                properties[ov.properties.hint.num_requests()] = int(self.config.num_streams)
            except ValueError:
                pass  # Use AUTO

        if self.config.num_threads > 0:
            properties[ov.properties.inference_num_threads()] = self.config.num_threads

        if self.config.bind_thread:
            properties[ov.properties.hint.enable_cpu_pinning()] = True

        if self.config.enable_profiling:
            properties[ov.properties.enable_profiling()] = True

        return properties

    def _build_accelerator_compile_properties(self) -> Dict[str, Any]:
        """Build accelerator-specific compilation properties.

        For accelerators, only use properties explicitly passed via -p/--properties.
        Default CPU hints like PERFORMANCE_HINT are not applied automatically.
        """
        properties = {}

        if self.config.enable_profiling:
            properties[ov.properties.enable_profiling()] = True

        # Add ONLY user-specified device properties from -p/--properties
        # Don't add default hints - accelerators may not support them
        if hasattr(self.config, 'device_properties') and self.config.device_properties:
            for key, value in self.config.device_properties.items():
                try:
                    if isinstance(value, str):
                        if value.isdigit():
                            properties[key] = int(value)
                        elif value.upper() in ('TRUE', 'FALSE'):
                            properties[key] = value.upper() == 'TRUE'
                        else:
                            properties[key] = value
                    else:
                        properties[key] = value
                except (AttributeError, ValueError):
                    properties[key] = value

        return properties

    def _reshape_model_for_batch(self, batch_size: int) -> None:
        """Reshape model inputs to the specified batch size."""
        new_shapes = {}
        for input_node in self._model.inputs:
            name = input_node.any_name
            current_shape = input_node.partial_shape

            new_dims = []
            for i, dim in enumerate(current_shape):
                if i == 0:
                    new_dims.append(batch_size)
                else:
                    if dim.is_static:
                        new_dims.append(dim.get_length())
                    else:
                        new_dims.append(-1)

            new_shapes[name] = new_dims

        self._model.reshape(new_shapes)
        logger.debug(f"Model reshaped for batch_size={batch_size}")

    def _apply_nhwc_input_layout(self) -> None:
        """Apply NHWC input layout using PrePostProcessor.

        This allows the model to accept NHWC (height, width, channels) input
        while the internal model uses NCHW layout.
        """
        from openvino.preprocess import PrePostProcessor

        ppp = PrePostProcessor(self._model)
        ppp.input().tensor().set_layout("NHWC")
        ppp.input().model().set_layout("NCHW")
        self._model = ppp.build()
        logger.debug("Applied NHWC input layout via PrePostProcessor")

    def _extract_model_info(self) -> None:
        """Extract input/output information from the model."""
        self._input_names = []
        self._input_shapes = {}
        self._input_dtypes = {}

        for input_node in self._model.inputs:
            name = input_node.any_name
            self._input_names.append(name)

            shape = tuple(input_node.partial_shape.get_min_shape())
            if any(d == 0 for d in shape):
                # Dynamic shape, use default
                shape = tuple(d if d > 0 else 1 for d in shape)

            self._input_shapes[name] = shape
            self._input_dtypes[name] = input_node.element_type

        self._output_names = []
        self._output_shapes = {}

        for output_node in self._model.outputs:
            name = output_node.any_name
            self._output_names.append(name)

            shape = tuple(output_node.partial_shape.get_min_shape())
            if any(d == 0 for d in shape):
                shape = tuple(d if d > 0 else 1 for d in shape)

            self._output_shapes[name] = shape

    def _create_infer_requests(self) -> None:
        """Create inference requests for async execution."""
        self._infer_requests = []

        nireq = max(1, self._optimal_nireq)
        for _ in range(nireq):
            request = self._compiled_model.create_infer_request()
            self._infer_requests.append(request)

        self._infer_request = self._infer_requests[0]

        self._request_queue = queue.Queue()
        for req in self._infer_requests:
            self._request_queue.put(req)

    def add_infer_requests(self, count: int) -> None:
        """Add more inference requests for higher parallelism (e.g., Server mode)."""
        if not self._loaded:
            self.load()

        for _ in range(count):
            request = self._compiled_model.create_infer_request()
            self._infer_requests.append(request)
            self._request_queue.put(request)

    def _convert_to_model_dtype(self, name: str, data: np.ndarray) -> np.ndarray:
        """Convert input data to the dtype expected by the model."""
        if name not in self._input_dtypes:
            return data

        element_type = self._input_dtypes[name]
        element_type_str = str(element_type)

        # Map OpenVINO element types to numpy dtypes
        if 'i64' in element_type_str or 'int64' in element_type_str.lower():
            return data.astype(np.int64)
        elif 'i32' in element_type_str or 'int32' in element_type_str.lower():
            return data.astype(np.int32)
        elif 'f32' in element_type_str or 'float32' in element_type_str.lower():
            return data.astype(np.float32)
        elif 'f16' in element_type_str or 'float16' in element_type_str.lower():
            return data.astype(np.float16)

        return data

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run synchronous inference (NOT thread-safe, use predict_threadsafe for multi-threaded)."""
        if not self._loaded:
            self.load()

        for name, data in inputs.items():
            converted_data = self._convert_to_model_dtype(name, data)
            self._infer_request.set_tensor(name, ov.Tensor(converted_data))

        self._infer_request.infer()

        outputs = {}
        for name in self._output_names:
            output_tensor = self._infer_request.get_tensor(name)
            outputs[name] = output_tensor.data.copy()

        return outputs

    def predict_threadsafe(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run synchronous inference with per-request locking for thread safety."""
        if not self._loaded:
            self.load()

        # Get an available request from queue (blocks if none available)
        request = self._request_queue.get()

        try:
            for name, data in inputs.items():
                converted_data = self._convert_to_model_dtype(name, data)
                request.set_tensor(name, ov.Tensor(converted_data))

            # Run inference (this blocks until complete)
            request.infer()

            outputs = {}
            for name in self._output_names:
                output_tensor = request.get_tensor(name)
                outputs[name] = output_tensor.data.copy()

            return outputs
        finally:
            self._request_queue.put(request)

    def predict_batch(self, batch: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Run inference on a batch of inputs using multiple inference requests."""
        if not self._loaded:
            self.load()

        results = []
        num_requests = len(self._infer_requests)

        for i in range(0, len(batch), num_requests):
            chunk = batch[i:i + num_requests]

            for j, inputs in enumerate(chunk):
                request = self._infer_requests[j]

                for name, data in inputs.items():
                    converted_data = self._convert_to_model_dtype(name, data)
                    request.set_tensor(name, ov.Tensor(converted_data))

                request.start_async()

            for j in range(len(chunk)):
                request = self._infer_requests[j]
                request.wait()

                outputs = {}
                for name in self._output_names:
                    output_tensor = request.get_tensor(name)
                    outputs[name] = output_tensor.data.copy()

                results.append(outputs)

        return results

    def predict_async(self, inputs: Dict[str, np.ndarray], request_id: int = 0) -> InferRequest:
        """Start asynchronous inference, returning the request to wait on."""
        if not self._loaded:
            self.load()

        request = self._infer_requests[request_id % len(self._infer_requests)]

        for name, data in inputs.items():
            converted_data = self._convert_to_model_dtype(name, data)
            request.set_tensor(name, ov.Tensor(converted_data))

        request.start_async()
        return request

    def get_results_async(self, request: InferRequest) -> Dict[str, np.ndarray]:
        """Wait for async request completion and return output tensors."""
        request.wait()

        outputs = {}
        for name in self._output_names:
            output_tensor = request.get_tensor(name)
            outputs[name] = output_tensor.data.copy()

        return outputs

    def create_async_queue(
        self,
        num_jobs: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> "AsyncInferQueue":
        """Create an AsyncInferQueue for high-throughput async inference."""
        if not self._loaded:
            self.load()

        if num_jobs is None:
            num_jobs = self._optimal_nireq

        # Ensure we have enough parallel jobs for high throughput
        num_jobs = max(num_jobs, self._optimal_nireq)

        async_queue = AsyncInferQueue(self._compiled_model, num_jobs)

        if callback:
            async_queue.set_callback(callback)

        return async_queue

    def start_async_queue(
        self,
        async_queue: "AsyncInferQueue",
        inputs: Dict[str, np.ndarray],
        userdata: Any = None
    ) -> None:
        """Start async inference on the queue."""
        converted = {}
        for name, data in inputs.items():
            converted[name] = self._convert_to_model_dtype(name, data)

        async_queue.start_async(converted, userdata)

    @property
    def input_names(self) -> List[str]:
        """Get list of input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Get list of output tensor names."""
        return self._output_names

    @property
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of input tensors."""
        return self._input_shapes

    @property
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of output tensors."""
        return self._output_shapes

    @property
    def num_streams(self) -> int:
        """Get optimal number of inference requests."""
        return self._optimal_nireq

    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = super().get_info()

        if OPENVINO_AVAILABLE:
            info["openvino_version"] = ov.__version__

        if self._loaded:
            info.update({
                "device": self.config.device,
                "optimal_nireq": self._optimal_nireq,
                "performance_hint": self.config.performance_hint,
                "inference_precision": self.config.inference_precision,
            })

            if self._core:
                try:
                    info["device_full_name"] = self._core.get_property(
                        self.config.device,
                        "FULL_DEVICE_NAME"
                    )
                except Exception:
                    pass

        return info

    def benchmark(self, num_iterations: int = 100, warmup_iterations: int = 10) -> Dict[str, float]:
        """Run a simple latency benchmark returning timing statistics in ms."""
        import time

        if not self._loaded:
            self.load()

        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            dummy_inputs[name] = np.random.randn(*shape).astype(np.float32)

        for _ in range(warmup_iterations):
            self.predict(dummy_inputs)

        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(dummy_inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000.0 / np.mean(latencies)),
        }
