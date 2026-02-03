"""3D-UNet System Under Test (SUT) implementation for MLPerf.

This module implements sliding window inference for 3D medical image
segmentation using OpenVINO, following the MLPerf nnU-Net reference.
"""

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
from ..datasets.kits19 import KiTS19QSL, NUM_CLASSES, PADDING_VAL

logger = logging.getLogger(__name__)


class UNet3DSUT:
    """System Under Test for 3D-UNet medical image segmentation.

    Performs sliding window inference with Gaussian-weighted patch
    aggregation, following the MLPerf nnU-Net reference implementation.
    """

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

        # Map I/O names
        self.input_name = config.model.input_name
        if self.input_name not in self.backend.input_names:
            self.input_name = self.backend.input_names[0]

        self.output_name = config.model.output_name
        if self.output_name not in self.backend.output_names:
            self.output_name = self.backend.output_names[0]

        # Sliding window parameters from dataset
        self._patch_size = qsl.dataset.patch_size
        self._gaussian_map = qsl.dataset.get_gaussian_map()

        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        self._sut = None
        self._qsl_handle = None

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
                logger.info(
                    f"Progress: {self._sample_count} samples, "
                    f"{throughput:.2f} samples/sec"
                )
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completed: {self._sample_count} samples in "
                f"{elapsed:.1f}s ({throughput:.2f} samples/sec)"
            )

    def _run_sliding_window(self, volume: np.ndarray) -> np.ndarray:
        """Run sliding window inference on a 3D volume.

        Args:
            volume: Preprocessed 3D volume of shape (D, H, W).

        Returns:
            Segmentation mask of shape (D, H, W) with class labels.
        """
        volume_shape = volume.shape
        patch_size = self._patch_size
        gaussian_map = self._gaussian_map

        # Accumulate predictions weighted by Gaussian
        output_shape = (NUM_CLASSES,) + volume_shape
        aggregated = np.zeros(output_shape, dtype=np.float32)
        weight_sum = np.zeros(volume_shape, dtype=np.float32)

        # Get patch positions
        positions = self.qsl.dataset.get_sliding_window_positions(volume_shape)

        for pos in positions:
            d, h, w = pos
            pd, ph, pw = patch_size

            # Extract patch
            patch = volume[d:d + pd, h:h + ph, w:w + pw]

            # Pad if necessary (for edge cases where volume < patch_size)
            actual_shape = patch.shape
            if actual_shape != patch_size:
                padded = np.full(patch_size, PADDING_VAL, dtype=np.float32)
                padded[:actual_shape[0], :actual_shape[1], :actual_shape[2]] = patch
                patch = padded

            # Format input: (1, 1, D, H, W)
            input_data = patch.reshape(1, 1, *patch_size).astype(np.float32)

            # Run inference
            outputs = self.backend.predict({self.input_name: input_data})
            output = outputs.get(self.output_name, list(outputs.values())[0])

            # output shape: (1, num_classes, D, H, W) or (num_classes, D, H, W)
            if output.ndim == 5:
                output = output[0]  # Remove batch dim -> (C, D, H, W)

            # Crop back to actual size if padded
            output = output[
                :,
                :actual_shape[0],
                :actual_shape[1],
                :actual_shape[2],
            ]
            crop_gaussian = gaussian_map[
                :actual_shape[0],
                :actual_shape[1],
                :actual_shape[2],
            ]

            # Accumulate with Gaussian weighting
            aggregated[:, d:d + actual_shape[0], h:h + actual_shape[1], w:w + actual_shape[2]] += (
                output * crop_gaussian[np.newaxis, ...]
            )
            weight_sum[d:d + actual_shape[0], h:h + actual_shape[1], w:w + actual_shape[2]] += crop_gaussian

        # Normalize by weights
        weight_sum = np.clip(weight_sum, a_min=1e-7, a_max=None)
        aggregated /= weight_sum[np.newaxis, ...]

        # Argmax to get class predictions (uint8 per MLPerf reference)
        segmentation = np.argmax(aggregated, axis=0).astype(np.uint8)

        return segmentation

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """Process a single sample (full volume with sliding window)."""
        features = self.qsl.get_features(sample_idx)
        volume = features["volume"]
        return self._run_sliding_window(volume)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries in Offline mode (MLPerf reference: base_SUT.py).

        Sends actual segmentation bytes per-sample for accuracy log
        compatibility with official accuracy_kits.py script.
        """
        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="3D-UNet Offline inference")

        for qs in query_samples:
            sample_idx = qs.index
            segmentation = self._process_sample(sample_idx)
            self._predictions[sample_idx] = segmentation

            # Send actual segmentation bytes (matches reference base_SUT.py)
            response_array = array.array("B", segmentation.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

            self._sample_count += 1
            self._update_progress(1)

        self._close_progress()
        self._query_count += 1

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries in Server mode."""
        if self._query_count == 0:
            self._start_progress(0, desc="3D-UNet Server inference")

        for qs in query_samples:
            sample_idx = qs.index
            segmentation = self._process_sample(sample_idx)
            self._predictions[sample_idx] = segmentation

            response_array = array.array("B", segmentation.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

            self._sample_count += 1
            self._update_progress(1)

        self._query_count += 1

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries from LoadGen."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut is None:
            self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
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
        return f"3D-UNet-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all predictions (segmentation masks)."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset SUT state."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mean Dice score accuracy."""
        if not self._predictions:
            logger.warning(
                f"No predictions (sample_count={self._sample_count}, "
                f"query_count={self._query_count})"
            )
            return {"mean_dice": 0.0, "num_samples": 0}

        predictions = []
        ground_truths = []

        for sample_idx in sorted(self._predictions.keys()):
            predictions.append(self._predictions[sample_idx])
            ground_truths.append(self.qsl.get_label(sample_idx))

        return self.qsl.dataset.compute_accuracy(predictions, ground_truths)
