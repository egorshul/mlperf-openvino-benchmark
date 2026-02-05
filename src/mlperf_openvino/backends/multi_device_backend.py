"""Multi-device backend for accelerators with multiple dies."""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .device_discovery import (
    discover_accelerator_devices,
    validate_accelerator_device,
    get_card_and_die,
    is_accelerator_die,
)
from ..core.config import OpenVINOConfig

logger = logging.getLogger(__name__)


class DieContext:
    """Context for a single accelerator die, holding compiled model and requests."""

    def __init__(
        self,
        device_name: str,
        compiled_model: CompiledModel,
        optimal_nireq: int,
    ):
        self.device_name = device_name
        self.compiled_model = compiled_model
        self.optimal_nireq = optimal_nireq

        self.infer_requests: List[InferRequest] = []
        self.request_queue: queue.Queue = queue.Queue()

        for _ in range(optimal_nireq):
            req = compiled_model.create_infer_request()
            self.infer_requests.append(req)
            self.request_queue.put(req)

        logger.info(f"Die {device_name}: created {optimal_nireq} inference requests")

    def get_request(self) -> InferRequest:
        return self.request_queue.get()

    def return_request(self, request: InferRequest) -> None:
        self.request_queue.put(request)


class MultiDeviceBackend(BaseBackend):
    """Backend for parallel inference across multiple accelerator dies."""

    def __init__(
        self,
        model_path: str,
        config: Optional[OpenVINOConfig] = None,
        target_devices: Optional[List[str]] = None,
        **kwargs
    ):
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO is not installed. Please install it with: "
                "pip install openvino"
            )

        super().__init__(model_path, **kwargs)

        self.config = config or OpenVINOConfig()
        self._target_devices = target_devices

        self._core: Optional[Core] = None
        self._model: Optional[ov.Model] = None

        self._die_contexts: Dict[str, DieContext] = {}
        self._active_devices: List[str] = []

        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        self._input_dtypes: Dict[str, Any] = {}

        self._executor: Optional[ThreadPoolExecutor] = None

        self._request_counter = 0
        self._request_lock = threading.Lock()

    def load(self) -> None:
        if self._loaded:
            logger.warning("Model already loaded, skipping...")
            return

        logger.info(f"Loading model from {self.model_path}")

        self._core = Core()

        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._core.set_property({"CACHE_DIR": str(cache_path)})

        device_prefix = self.config.get_device_prefix()
        logger.info(f"Device prefix: {device_prefix}")

        if self._target_devices:
            self._active_devices = self._validate_devices(self._target_devices)
        else:
            self._active_devices = discover_accelerator_devices(self._core, device_prefix)

        if not self._active_devices:
            raise RuntimeError(
                f"No {device_prefix} devices available for inference. "
                f"Available devices: {self._core.available_devices}"
            )

        logger.info(f"Will use {device_prefix} dies: {self._active_devices}")

        model_path = Path(self.model_path)
        if model_path.suffix.lower() in (".onnx", ".xml"):
            self._model = self._core.read_model(str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        self._reshape_model_for_batch(max(self.config.batch_size, 1))
        self._extract_model_info()

        properties = self._build_compile_properties()

        for device_name in self._active_devices:
            self._compile_for_die(device_name, properties)

        num_workers = len(self._active_devices) * 2
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        self._loaded = True
        logger.info(
            f"Multi-device backend loaded: {len(self._active_devices)} dies, "
            f"total {self.total_infer_requests} inference requests"
        )

    def _validate_devices(self, devices: List[str]) -> List[str]:
        valid_devices = []
        device_prefix = self.config.get_device_prefix()

        for device in devices:
            is_valid, error = validate_accelerator_device(self._core, device)
            if is_valid:
                if is_accelerator_die(device, device_prefix):
                    valid_devices.append(device)
                elif device.upper() == device_prefix:
                    valid_devices.extend(discover_accelerator_devices(self._core, device_prefix))
            else:
                logger.warning(f"Skipping invalid device: {error}")

        return list(dict.fromkeys(valid_devices))

    def _compile_for_die(self, device_name: str, properties: Dict[str, Any]) -> None:
        logger.info(f"Compiling model for {device_name}...")

        try:
            compiled_model = self._core.compile_model(
                self._model,
                device_name,
                properties
            )

            try:
                optimal_nireq = compiled_model.get_property(
                    ov.properties.optimal_number_of_infer_requests()
                )
            except Exception:
                optimal_nireq = 4
                logger.warning(f"{device_name}: Could not get optimal_nireq, using {optimal_nireq}")

            self._die_contexts[device_name] = DieContext(
                device_name=device_name,
                compiled_model=compiled_model,
                optimal_nireq=optimal_nireq,
            )

            logger.info(f"{device_name}: Compiled successfully (optimal_nireq={optimal_nireq})")

        except Exception as e:
            logger.error(f"Failed to compile model for {device_name}: {e}")
            raise

    def _build_compile_properties(self) -> Dict[str, Any]:
        """For accelerators, only use properties explicitly passed via -p/--properties."""
        properties = {}

        if self.config.enable_profiling:
            properties[ov.properties.enable_profiling()] = True

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

        if properties:
            logger.info(f"Accelerator compile properties: {properties}")
        else:
            logger.info("No custom compile properties specified for accelerator")

        return properties

    def _reshape_model_for_batch(self, batch_size: int) -> None:
        logger.info(f"Reshaping model for batch_size={batch_size}")

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
            logger.info(f"  {name}: {current_shape} -> {new_dims}")

        self._model.reshape(new_shapes)
        logger.info(f"Model reshaped successfully for batch_size={batch_size}")

    def _extract_model_info(self) -> None:
        self._input_names = []
        self._input_shapes = {}
        self._input_dtypes = {}

        for input_node in self._model.inputs:
            name = input_node.any_name
            self._input_names.append(name)

            shape = tuple(input_node.partial_shape.get_min_shape())
            if any(d == 0 for d in shape):
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

        logger.info(f"Model inputs: {self._input_names}")
        logger.info(f"Model outputs: {self._output_names}")

    def _convert_to_model_dtype(self, name: str, data: np.ndarray) -> np.ndarray:
        if name not in self._input_dtypes:
            return data

        element_type = self._input_dtypes[name]
        element_type_str = str(element_type)

        if 'i64' in element_type_str or 'int64' in element_type_str.lower():
            return data.astype(np.int64)
        elif 'i32' in element_type_str or 'int32' in element_type_str.lower():
            return data.astype(np.int32)
        elif 'f32' in element_type_str or 'float32' in element_type_str.lower():
            return data.astype(np.float32)
        elif 'f16' in element_type_str or 'float16' in element_type_str.lower():
            return data.astype(np.float16)

        return data

    def _get_next_die(self) -> str:
        with self._request_lock:
            die = self._active_devices[self._request_counter % len(self._active_devices)]
            self._request_counter += 1
        return die

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._loaded:
            self.load()

        die_name = self._get_next_die()
        ctx = self._die_contexts[die_name]

        request = ctx.get_request()

        try:
            for name, data in inputs.items():
                converted = self._convert_to_model_dtype(name, data)
                request.set_tensor(name, ov.Tensor(converted))

            request.infer()

            outputs = {}
            for name in self._output_names:
                output_tensor = request.get_tensor(name)
                outputs[name] = output_tensor.data.copy()

            return outputs

        finally:
            ctx.return_request(request)

    def predict_batch(
        self,
        batch: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """Run batch inference across all dies in parallel, returning results in input order."""
        if not self._loaded:
            self.load()

        if not batch:
            return []

        results = [None] * len(batch)

        def process_sample(idx: int, inputs: Dict[str, np.ndarray]) -> Tuple[int, Dict]:
            return idx, self.predict(inputs)

        futures = []
        for idx, inputs in enumerate(batch):
            future = self._executor.submit(process_sample, idx, inputs)
            futures.append(future)

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

        return results

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        return self._output_names

    @property
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._input_shapes

    @property
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._output_shapes

    @property
    def num_streams(self) -> int:
        return self.total_infer_requests

    @property
    def total_infer_requests(self) -> int:
        total = 0
        for ctx in self._die_contexts.values():
            total += ctx.optimal_nireq
        return total

    @property
    def active_devices(self) -> List[str]:
        return self._active_devices.copy()

    @property
    def num_dies(self) -> int:
        return len(self._active_devices)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "backend_type": "multi_device_accelerator",
            "active_devices": self._active_devices,
            "num_dies": self.num_dies,
            "total_infer_requests": self.total_infer_requests,
        })

        if self._loaded:
            info["performance_hint"] = self.config.performance_hint
            info["device_properties"] = getattr(self.config, 'device_properties', {})

            dies_info = {}
            for die_name, ctx in self._die_contexts.items():
                card, die_on_card = get_card_and_die(die_name) or (None, None)
                dies_info[die_name] = {
                    "optimal_nireq": ctx.optimal_nireq,
                    "card_index": card,
                    "die_on_card": die_on_card,
                }
            info["dies"] = dies_info

        return info

    def unload(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._die_contexts.clear()
        self._active_devices.clear()
        self._model = None
        self._core = None
        self._loaded = False

        logger.info("Multi-device backend unloaded")

    def __del__(self):
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)
