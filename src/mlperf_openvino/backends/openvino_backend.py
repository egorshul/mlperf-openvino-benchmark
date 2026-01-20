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
        **kwargs
    ):
        """
        Initialize OpenVINO backend.

        Args:
            model_path: Path to ONNX or OpenVINO IR model
            config: OpenVINO configuration
            **kwargs: Additional options
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO is not installed. Please install it with: "
                "pip install openvino"
            )

        super().__init__(model_path, **kwargs)

        self.config = config or OpenVINOConfig()
        self._core: Optional[Core] = None
        self._model: Optional[ov.Model] = None
        self._compiled_model: Optional[CompiledModel] = None
        self._infer_request: Optional[InferRequest] = None

        # For async inference
        self._infer_requests: List[InferRequest] = []
        self._optimal_nireq: int = 1  # Optimal number of inference requests

        # For thread-safe inference - queue of available requests
        self._request_queue: Optional[queue.Queue] = None

        # Cache input/output info
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

        # Initialize OpenVINO Core
        self._core = Core()

        # Create cache directory if specified
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._core.set_property({"CACHE_DIR": str(cache_path)})

        # Load the model
        model_path = Path(self.model_path)

        if model_path.suffix.lower() == ".onnx":
            logger.info("Loading ONNX model and converting to OpenVINO IR...")
            self._model = self._core.read_model(str(model_path))
        elif model_path.suffix.lower() == ".xml":
            logger.info("Loading OpenVINO IR model...")
            self._model = self._core.read_model(str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        # Get model info before compilation
        self._extract_model_info()

        # Compile the model with optimizations
        logger.info(f"Compiling model for device: {self.config.device}")

        # Build properties
        properties = self._build_compile_properties()

        self._compiled_model = self._core.compile_model(
            self._model,
            self.config.device,
            properties
        )

        # Get optimal number of inference requests - KEY for performance!
        try:
            self._optimal_nireq = self._compiled_model.get_property(
                ov.properties.optimal_number_of_infer_requests()
            )
            logger.info(f"Optimal number of inference requests: {self._optimal_nireq}")
        except Exception:
            self._optimal_nireq = 4  # Fallback
            logger.warning(f"Could not get optimal nireq, using {self._optimal_nireq}")

        # Create inference requests
        self._create_infer_requests()

        self._loaded = True
        logger.info("Model loaded and compiled successfully")
    
    def _build_compile_properties(self) -> Dict[str, Any]:
        """Build compilation properties from config."""
        properties = {}

        # Use device-specific property builders from config
        if self.config.is_accelerator_device():
            return self._build_accelerator_compile_properties()
        else:
            return self._build_cpu_compile_properties()

    def _build_cpu_compile_properties(self) -> Dict[str, Any]:
        """Build CPU-specific compilation properties."""
        properties = {}

        # Performance hint
        if self.config.performance_hint:
            hint_enum = getattr(ov.properties.hint.PerformanceMode,
                              self.config.performance_hint, None)
            if hint_enum:
                properties[ov.properties.hint.performance_mode()] = hint_enum

        # Number of streams
        if self.config.num_streams != "AUTO":
            try:
                properties[ov.properties.hint.num_requests()] = int(self.config.num_streams)
            except ValueError:
                pass  # Use AUTO

        # Number of threads
        if self.config.num_threads > 0:
            properties[ov.properties.inference_num_threads()] = self.config.num_threads

        # CPU-specific options
        if self.config.bind_thread:
            properties[ov.properties.hint.enable_cpu_pinning()] = True

        # Enable profiling if requested
        if self.config.enable_profiling:
            properties[ov.properties.enable_profiling()] = True

        return properties

    def _build_accelerator_compile_properties(self) -> Dict[str, Any]:
        """Build accelerator-specific compilation properties."""
        properties = {}

        # Performance hint
        if self.config.performance_hint:
            hint_enum = getattr(ov.properties.hint.PerformanceMode,
                              self.config.performance_hint, None)
            if hint_enum:
                properties[ov.properties.hint.performance_mode()] = hint_enum

        # Number of streams
        if self.config.num_streams != "AUTO":
            try:
                properties[ov.properties.hint.num_requests()] = int(self.config.num_streams)
            except ValueError:
                pass  # Use AUTO

        # Enable profiling if requested
        if self.config.enable_profiling:
            properties[ov.properties.enable_profiling()] = True

        # Add user-specified device properties for accelerator
        if hasattr(self.config, 'device_properties') and self.config.device_properties:
            for key, value in self.config.device_properties.items():
                # Try to convert types
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
    
    def _extract_model_info(self) -> None:
        """Extract input/output information from the model."""
        # Get input info
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

            # Get element type
            element_type = input_node.element_type
            self._input_dtypes[name] = element_type

        # Get output info
        self._output_names = []
        self._output_shapes = {}

        for output_node in self._model.outputs:
            name = output_node.any_name
            self._output_names.append(name)

            shape = tuple(output_node.partial_shape.get_min_shape())
            if any(d == 0 for d in shape):
                shape = tuple(d if d > 0 else 1 for d in shape)

            self._output_shapes[name] = shape

        logger.info(f"Model inputs: {self._input_names}")
        logger.info(f"Model input dtypes: {self._input_dtypes}")
        logger.info(f"Model outputs: {self._output_names}")
    
    def _create_infer_requests(self) -> None:
        """Create inference requests for async execution."""
        self._infer_requests = []

        # Create optimal number of requests for maximum throughput
        nireq = max(1, self._optimal_nireq)
        for _ in range(nireq):
            request = self._compiled_model.create_infer_request()
            self._infer_requests.append(request)

        # Set default request for sync inference
        self._infer_request = self._infer_requests[0]

        # Create queue for thread-safe inference
        self._request_queue = queue.Queue()
        for req in self._infer_requests:
            self._request_queue.put(req)

        logger.info(f"Created {nireq} inference requests")

    def add_infer_requests(self, count: int) -> None:
        """Add more inference requests for higher parallelism (e.g., Server mode)."""
        if not self._loaded:
            self.load()

        for _ in range(count):
            request = self._compiled_model.create_infer_request()
            self._infer_requests.append(request)
            self._request_queue.put(request)

        logger.info(f"Added {count} inference requests, total: {len(self._infer_requests)}")
    
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
        """
        Run synchronous inference (NOT thread-safe, use predict_threadsafe for multi-threaded).

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not self._loaded:
            self.load()

        # Set input tensors with automatic dtype conversion
        for name, data in inputs.items():
            converted_data = self._convert_to_model_dtype(name, data)
            self._infer_request.set_tensor(name, ov.Tensor(converted_data))

        # Run inference
        self._infer_request.infer()

        # Get output tensors
        outputs = {}
        for name in self._output_names:
            output_tensor = self._infer_request.get_tensor(name)
            outputs[name] = output_tensor.data.copy()

        return outputs

    def predict_threadsafe(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run synchronous inference in a thread-safe manner.

        Uses a queue of inference requests - each thread gets exclusive
        access to a request while running inference.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not self._loaded:
            self.load()

        # Get an available request from queue (blocks if none available)
        request = self._request_queue.get()

        try:
            # Set input tensors with automatic dtype conversion
            for name, data in inputs.items():
                converted_data = self._convert_to_model_dtype(name, data)
                request.set_tensor(name, ov.Tensor(converted_data))

            # Run inference (this blocks until complete)
            request.infer()

            # Get output tensors
            outputs = {}
            for name in self._output_names:
                output_tensor = request.get_tensor(name)
                outputs[name] = output_tensor.data.copy()

            return outputs
        finally:
            # Return request to queue for reuse
            self._request_queue.put(request)
    
    def predict_batch(
        self, 
        batch: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on a batch of inputs.
        
        For optimal throughput, this method uses multiple inference requests.
        
        Args:
            batch: List of input dictionaries
            
        Returns:
            List of output dictionaries
        """
        if not self._loaded:
            self.load()
        
        results = []
        num_requests = len(self._infer_requests)
        
        # Process in chunks matching the number of available requests
        for i in range(0, len(batch), num_requests):
            chunk = batch[i:i + num_requests]
            
            # Start all inferences
            for j, inputs in enumerate(chunk):
                request = self._infer_requests[j]

                for name, data in inputs.items():
                    converted_data = self._convert_to_model_dtype(name, data)
                    request.set_tensor(name, ov.Tensor(converted_data))

                request.start_async()
            
            # Wait for all to complete
            for j in range(len(chunk)):
                request = self._infer_requests[j]
                request.wait()
                
                outputs = {}
                for name in self._output_names:
                    output_tensor = request.get_tensor(name)
                    outputs[name] = output_tensor.data.copy()
                
                results.append(outputs)
        
        return results
    
    def predict_async(
        self,
        inputs: Dict[str, np.ndarray],
        request_id: int = 0
    ) -> InferRequest:
        """
        Start asynchronous inference.
        
        Args:
            inputs: Input data
            request_id: ID of the inference request to use
            
        Returns:
            The inference request (can be used to wait for completion)
        """
        if not self._loaded:
            self.load()
        
        request = self._infer_requests[request_id % len(self._infer_requests)]

        for name, data in inputs.items():
            converted_data = self._convert_to_model_dtype(name, data)
            request.set_tensor(name, ov.Tensor(converted_data))

        request.start_async()
        return request
    
    def get_results_async(self, request: InferRequest) -> Dict[str, np.ndarray]:
        """
        Get results from async inference request.
        
        Args:
            request: The inference request
            
        Returns:
            Output dictionary
        """
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
        """
        Create an AsyncInferQueue for high-throughput async inference.

        Args:
            num_jobs: Number of parallel jobs (default: optimal_nireq)
            callback: Callback function called when inference completes.
                      Signature: callback(infer_request, userdata)

        Returns:
            AsyncInferQueue instance
        """
        if not self._loaded:
            self.load()

        if num_jobs is None:
            num_jobs = self._optimal_nireq

        # Ensure we have enough parallel jobs for high throughput
        num_jobs = max(num_jobs, self._optimal_nireq)

        async_queue = AsyncInferQueue(self._compiled_model, num_jobs)

        if callback:
            async_queue.set_callback(callback)

        logger.info(f"Created AsyncInferQueue with {num_jobs} parallel jobs (optimal_nireq={self._optimal_nireq})")
        return async_queue

    def start_async_queue(
        self,
        async_queue: "AsyncInferQueue",
        inputs: Dict[str, np.ndarray],
        userdata: Any = None
    ) -> None:
        """
        Start async inference on the queue.

        Args:
            async_queue: The AsyncInferQueue
            inputs: Input tensors
            userdata: User data passed to callback
        """
        # Convert inputs to proper dtypes
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
            
            # Get device info
            if self._core:
                try:
                    info["device_full_name"] = self._core.get_property(
                        self.config.device, 
                        "FULL_DEVICE_NAME"
                    )
                except Exception:
                    pass
        
        return info
    
    def benchmark(
        self, 
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Run a simple latency benchmark.
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results (latency in ms)
        """
        import time
        
        if not self._loaded:
            self.load()
        
        # Create dummy input
        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            dummy_inputs[name] = np.random.randn(*shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            self.predict(dummy_inputs)
        
        # Benchmark
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
