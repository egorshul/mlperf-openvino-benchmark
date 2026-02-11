"""3D UNET SUT with sliding window inference for single-device."""

import array
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, Scenario
from ..backends.base import BaseBackend
from ..datasets.kits19 import (
    KiTS19QSL,
    ROI_SHAPE,
    compute_sliding_window_positions,
    get_gaussian_importance_map,
)

logger = logging.getLogger(__name__)


class UNet3DSUT:
    """3D UNET SUT with sliding window inference for KiTS 2019."""

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: KiTS19QSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario

        if not self.backend.is_loaded:
            self.backend.load()

        self._map_io_names()

        self._gaussian_map = get_gaussian_importance_map(ROI_SHAPE)
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        self._sut = None
        self._qsl_handle = None

    def _map_io_names(self) -> None:
        model_inputs = self.backend.input_names
        self.input_name = None
        for name in model_inputs:
            name_lower = name.lower()
            if any(p in name_lower for p in ['input', 'image', 'data']):
                self.input_name = name
                break
        if not self.input_name and model_inputs:
            self.input_name = model_inputs[0]

        model_outputs = self.backend.output_names
        self.output_name = None
        for name in model_outputs:
            name_lower = name.lower()
            if any(p in name_lower for p in ['output', 'logits', 'prediction']):
                self.output_name = name
                break
        if not self.output_name and model_outputs:
            self.output_name = model_outputs[0]

    def _sliding_window_inference(self, volume: np.ndarray) -> np.ndarray:
        """Run sliding window inference on a single volume (C, D, H, W)."""
        spatial_shape = volume.shape[1:]  # (D, H, W)
        positions = compute_sliding_window_positions(spatial_shape)

        first_slices = positions[0]
        sub_vol = volume[:, first_slices[0], first_slices[1], first_slices[2]]
        sub_vol_batch = sub_vol[np.newaxis, ...]
        inputs = {self.input_name: sub_vol_batch.astype(np.float32)}
        output = self.backend.predict(inputs)
        first_result = output.get(self.output_name, list(output.values())[0])
        if first_result.ndim == 5:
            first_result = first_result[0]

        num_classes = first_result.shape[0]
        accumulator = np.zeros((num_classes, *spatial_shape), dtype=np.float32)
        weight_map = np.zeros(spatial_shape, dtype=np.float32)

        accumulator[:, first_slices[0], first_slices[1], first_slices[2]] += (
            first_result * self._gaussian_map[np.newaxis, ...]
        )
        weight_map[first_slices[0], first_slices[1], first_slices[2]] += self._gaussian_map

        for slices in positions[1:]:
            sub_vol = volume[:, slices[0], slices[1], slices[2]]
            sub_vol_batch = sub_vol[np.newaxis, ...]
            inputs = {self.input_name: sub_vol_batch.astype(np.float32)}
            output = self.backend.predict(inputs)
            result = output.get(self.output_name, list(output.values())[0])
            if result.ndim == 5:
                result = result[0]

            accumulator[:, slices[0], slices[1], slices[2]] += (
                result * self._gaussian_map[np.newaxis, ...]
            )
            weight_map[slices[0], slices[1], slices[2]] += self._gaussian_map

        weight_map = np.maximum(weight_map, 1e-8)
        accumulator /= weight_map[np.newaxis, ...]

        return accumulator

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """Process a single sample through sliding window inference."""
        features = self.qsl.get_features(sample_idx)
        volume = features["input"]
        if volume.ndim == 5:
            volume = volume[0]

        logits = self._sliding_window_inference(volume)
        segmentation = np.argmax(logits, axis=0).astype(np.uint8)

        return segmentation

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total, desc=desc, unit="samples",
                file=sys.stderr, dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="3D-UNET Offline inference")

        for qs in query_samples:
            sample_idx = qs.index
            segmentation = self._process_sample(sample_idx)
            self._predictions[sample_idx] = segmentation

            response_data = segmentation.tobytes()
            response_array = array.array('B', response_data)
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            responses.append(response)

            self._sample_count += 1
            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def issue_queries(self, query_samples: List[Any]) -> None:
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            raise ValueError(f"3D UNET only supports Offline scenario, got: {self.scenario}")

    def flush_queries(self) -> None:
        if self._progress_bar is not None:
            self._close_progress()

    def get_sut(self) -> Any:
        if self._sut is None:
            self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    @property
    def name(self) -> str:
        return f"3D-UNET-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, np.ndarray]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mean Dice score over all predictions."""
        if not self._predictions:
            logger.warning("No predictions for accuracy computation")
            return {"mean_dice": 0.0, "kidney_dice": 0.0, "tumor_dice": 0.0, "num_samples": 0}

        predictions = []
        labels = []
        for sample_idx in sorted(self._predictions.keys()):
            predictions.append(self._predictions[sample_idx])
            label = self.qsl.get_label(sample_idx)
            labels.append(label)

        return self.qsl.dataset.compute_accuracy(predictions, labels)
