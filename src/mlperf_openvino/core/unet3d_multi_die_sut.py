"""3D UNET multi-die SUT with C++ acceleration."""

import logging
import sys
import time
from typing import Any, Dict, List

import numpy as np

try:
    from ..cpp import UNet3DMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and UNet3DMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    UNet3DMultiDieCppSUT = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .image_multi_die_sut_base import ImageMultiDieSUTBase
from .config import BenchmarkConfig, Scenario
from ..datasets.base import QuerySampleLibrary
from ..datasets.kits19 import (
    ROI_SHAPE,
    compute_sliding_window_positions,
    get_gaussian_importance_map,
)

logger = logging.getLogger(__name__)


class UNet3DMultiDieCppSUTWrapper(ImageMultiDieSUTBase):
    """Multi-die C++ SUT wrapper for 3D UNET with sliding window inference."""

    MODEL_NAME = "3D-UNET"
    DEFAULT_OFFLINE_BATCH_SIZE = 1
    DEFAULT_OFFLINE_NIREQ_MULTIPLIER = 1
    DEFAULT_SERVER_NIREQ_MULTIPLIER = 2
    DEFAULT_EXPLICIT_BATCH_SIZE = 1
    DEFAULT_BATCH_TIMEOUT_US = 2000
    BATCH_SERVER_ACCURACY = False

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: "KiTS19QSL",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        self._gaussian_map = get_gaussian_importance_map(ROI_SHAPE)
        self._segmentation_predictions: Dict[int, np.ndarray] = {}
        super().__init__(config, qsl, scenario)

    def supports_native_benchmark(self) -> bool:
        return False

    def _check_cpp_availability(self) -> None:
        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "3D UNET C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

    def _create_cpp_sut(
        self,
        model_path: str,
        device_prefix: str,
        batch_size: int,
        compile_props: Dict,
        use_nhwc: bool,
        nireq_multiplier: int,
    ) -> Any:
        return UNet3DMultiDieCppSUT(
            model_path,
            device_prefix,
            batch_size,
            compile_props,
            False,
            nireq_multiplier,
        )

    def _sliding_window_inference(self, volume: np.ndarray, sample_idx: int) -> np.ndarray:
        """Run sliding window inference dispatching sub-volumes to C++ dies."""
        spatial_shape = volume.shape[1:]
        positions = compute_sliding_window_positions(spatial_shape)
        num_positions = len(positions)

        self._cpp_sut.reset_counters()
        self._cpp_sut.clear_predictions()
        self._cpp_sut.enable_direct_loadgen(False)
        self._cpp_sut.set_store_predictions(True)

        dummy_query_ids = list(range(num_positions))
        dummy_sample_indices = list(range(num_positions))

        for pos_idx, slices in enumerate(positions):
            sub_vol = volume[:, slices[0], slices[1], slices[2]]
            sub_vol_batch = sub_vol[np.newaxis, ...].astype(np.float32)
            if not sub_vol_batch.flags['C_CONTIGUOUS']:
                sub_vol_batch = np.ascontiguousarray(sub_vol_batch)

            self._cpp_sut.start_async_batch(
                sub_vol_batch,
                [dummy_query_ids[pos_idx]],
                [dummy_sample_indices[pos_idx]],
                1,
            )

        self._cpp_sut.wait_all()
        preds = self._cpp_sut.get_predictions()

        received = len(preds)
        if received < num_positions:
            logger.warning(
                "C++ SUT returned %d/%d predictions for sample %d "
                "(check for unsupported output type warnings above)",
                received, num_positions, sample_idx,
            )

        first_pred = preds.get(0)
        if first_pred is None:
            logger.error("No predictions from C++ SUT for sample %d", sample_idx)
            return np.zeros((3, *spatial_shape), dtype=np.float32)

        first_output = np.array(first_pred, dtype=np.float32)
        if first_output.ndim == 1:
            num_classes = first_output.size // (ROI_SHAPE[0] * ROI_SHAPE[1] * ROI_SHAPE[2])
            first_output = first_output.reshape(num_classes, *ROI_SHAPE)
        elif first_output.ndim == 4:
            num_classes = first_output.shape[0]
        elif first_output.ndim == 5:
            first_output = first_output[0]
            num_classes = first_output.shape[0]
        else:
            num_classes = 3

        accumulator = np.zeros((num_classes, *spatial_shape), dtype=np.float32)
        weight_map = np.zeros(spatial_shape, dtype=np.float32)

        for pos_idx, slices in enumerate(positions):
            pred = preds.get(pos_idx)
            if pred is None:
                continue

            output = np.array(pred, dtype=np.float32)
            if output.ndim == 1:
                output = output.reshape(num_classes, *ROI_SHAPE)
            elif output.ndim == 5:
                output = output[0]

            accumulator[:, slices[0], slices[1], slices[2]] += (
                output * self._gaussian_map[np.newaxis, ...]
            )
            weight_map[slices[0], slices[1], slices[2]] += self._gaussian_map

        weight_map = np.maximum(weight_map, 1e-8)
        accumulator /= weight_map[np.newaxis, ...]

        return accumulator

    def _issue_query_offline(self, query_samples: List) -> None:
        """Override offline dispatch to handle sliding window per sample."""
        import array as array_module

        self._start_time = time.time()
        num_samples = len(query_samples)

        print(f"[3D-UNET Offline] {num_samples} samples, sliding window inference", file=sys.stderr)

        responses = []
        response_arrays = []

        for idx, qs in enumerate(query_samples):
            sample_idx = qs.index

            features = self.qsl.get_features(sample_idx)
            volume = features["input"]
            if volume.ndim == 5:
                volume = volume[0]

            logits = self._sliding_window_inference(volume, sample_idx)
            segmentation = np.argmax(logits, axis=0).astype(np.uint8)

            self._segmentation_predictions[sample_idx] = segmentation

            response_data = segmentation.tobytes()
            response_array = array_module.array('B', response_data)
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))

            elapsed = time.time() - self._start_time
            print(
                f"\r[3D-UNET] {idx+1}/{num_samples} samples, "
                f"{elapsed:.1f}s elapsed",
                end="", file=sys.stderr, flush=True,
            )

        print(file=sys.stderr)
        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def issue_queries(self, query_samples: List) -> None:
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            raise ValueError(f"3D UNET only supports Offline scenario, got: {self.scenario}")

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load C++ SUT and reset segmentation storage."""
        self._segmentation_predictions.clear()
        super().load(is_accuracy_mode=is_accuracy_mode)

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Return segmentation predictions (not raw C++ predictions)."""
        return self._segmentation_predictions.copy()

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mean Dice score over all predictions."""
        predictions = self.get_predictions()

        if not predictions:
            logger.warning("No predictions for accuracy computation")
            return {"mean_dice": 0.0, "kidney_dice": 0.0, "tumor_dice": 0.0, "num_samples": 0}

        pred_list = []
        label_list = []
        for sample_idx in sorted(predictions.keys()):
            pred_list.append(predictions[sample_idx])
            label = self.qsl.get_label(sample_idx)
            label_list.append(label)

        return self.qsl.dataset.compute_accuracy(pred_list, label_list)


def is_unet3d_multi_die_cpp_available() -> bool:
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
