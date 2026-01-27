"""Python wrapper for C++ RetinaNetMultiDieCppSUT."""

import array
import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    from ..cpp import RetinaNetMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and RetinaNetMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    RetinaNetMultiDieCppSUT = None

from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class RetinaNetMultiDieCppSUTWrapper:
    """
    Wrapper for C++ multi-die RetinaNet SUT with MLPerf LoadGen integration.
    Uses AUTO_BATCH for Server mode batching.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen not installed")

        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Batch size:
        # - Server: 1 (AUTO_BATCH or explicit batching handles it)
        # - Offline: use configured batch_size, default to 2 for NPU (efficiency)
        if scenario == Scenario.SERVER:
            self.batch_size = 1
        else:
            configured_batch = config.openvino.batch_size
            # For NPU Offline, use at least batch_size=2 for better efficiency
            # RetinaNet is memory-heavy (800x800x3 = 7.7MB), so don't use too large batches
            if configured_batch <= 1:
                self.batch_size = 2  # Default for NPU Offline with RetinaNet
                logger.info(f"RetinaNet NPU Offline: using batch_size={self.batch_size} for efficiency")
            else:
                self.batch_size = configured_batch

        # Get input name from config
        self.input_name = config.model.input_name

        # Build compile properties
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = dict(config.openvino.device_properties)

        # Check if using NHWC input layout (default is NHWC)
        use_nhwc = True
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

        # Get nireq_multiplier from config
        # RetinaNet uses smaller values than ResNet due to large input (800x800x3 = 7.7MB)
        # ResNet: Server=2, Offline=4
        # RetinaNet: Server=2, Offline=2 (reduced due to memory)
        server_config = config.model.server if hasattr(config.model, 'server') else None
        if scenario == Scenario.SERVER:
            nireq_multiplier = getattr(server_config, 'nireq_multiplier', 2) if server_config else 2
        else:
            nireq_multiplier = 2  # Lower than ResNet (4) due to large input size

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = RetinaNetMultiDieCppSUT(
            config.model.model_path,
            device_prefix,
            self.batch_size,
            compile_props,
            use_nhwc,
            nireq_multiplier
        )

        # Set specific target devices if a specific die is selected (e.g., NPU.0)
        if config.openvino.is_specific_die():
            target_devices = [config.openvino.device]
            self._cpp_sut.set_target_devices(target_devices)
            logger.debug(f"Set target devices: {target_devices}")

        # Statistics
        self._start_time = 0.0
        self._query_count = 0
        self._server_callback_set = False
        self._qsl_validated = False

        logger.debug(f"RetinaNetMultiDieCppSUTWrapper: device_prefix={device_prefix}, batch_size={self.batch_size}")

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load and compile the model."""
        # Configure explicit batching if enabled (before load)
        # RetinaNet uses smaller batch (2) and longer timeout (1000us) than ResNet (4, 500us)
        # due to larger input size (800x800x3 = 7.7MB vs 224x224x3 = 0.6MB)
        if self.scenario == Scenario.SERVER:
            server_config = self.config.model.server if hasattr(self.config.model, 'server') else None
            if server_config and getattr(server_config, 'explicit_batching', False):
                batch_size = getattr(server_config, 'explicit_batch_size', 2)    # Smaller for RetinaNet
                timeout_us = getattr(server_config, 'batch_timeout_us', 1000)    # Longer for large input
                self._cpp_sut.enable_explicit_batching(True, batch_size, timeout_us)

        self._cpp_sut.load()
        self._cpp_sut.set_store_predictions(is_accuracy_mode)
        self._is_accuracy_mode = is_accuracy_mode
        self.input_name = self._cpp_sut.get_input_name()

        logger.info(
            f"C++ RetinaNet SUT loaded: {self._cpp_sut.get_num_dies()} dies, "
            f"{self._cpp_sut.get_total_requests()} total requests"
        )

    def warmup(self, iterations: int = 2) -> None:
        """Run warmup inferences on all dies."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded, call load() first")
        self._cpp_sut.warmup(iterations)

    @property
    def is_loaded(self) -> bool:
        return self._cpp_sut.is_loaded()

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _prevalidate_qsl_data(self) -> None:
        """Register QSL data pointers in C++ for fast Server mode dispatch."""
        self._qsl_data_ready = False
        self._cpp_qsl_registered = False

        if not hasattr(self.qsl, '_loaded_samples'):
            logger.warning("QSL does not have _loaded_samples")
            return

        if not self.qsl._loaded_samples:
            return

        first_idx = next(iter(self.qsl._loaded_samples.keys()))
        sample = self.qsl._loaded_samples[first_idx]

        if not isinstance(sample, np.ndarray):
            return

        needs_batch_dim = (sample.ndim == 3)
        needs_dtype_conv = (sample.dtype != np.float32)
        needs_contiguous = (not sample.flags['C_CONTIGUOUS'])

        if needs_dtype_conv or needs_contiguous:
            return

        try:
            registered_count = 0
            for idx, data in self.qsl._loaded_samples.items():
                if needs_batch_dim:
                    data = data[np.newaxis, ...]
                    if not data.flags['C_CONTIGUOUS']:
                        data = np.ascontiguousarray(data)
                    self.qsl._loaded_samples[idx] = data

                self._cpp_sut.register_sample_data(idx, data)
                registered_count += 1

            self._cpp_qsl_registered = True
            self._qsl_data_ready = True
            logger.debug(f"Registered {registered_count} samples in C++")
        except Exception as e:
            logger.debug(f"Failed to register samples: {e}")
            self._cpp_qsl_registered = False

    def _get_input_data(self, sample_idx: int) -> np.ndarray:
        """Get input data for a sample."""
        if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
            return self.qsl._loaded_samples[sample_idx]
        else:
            features = self.qsl.get_features(sample_idx)
            return features.get("input", features.get(self.input_name))

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode with batching."""
        import sys

        self._start_time = time.time()
        self._cpp_sut.reset_counters()
        self._cpp_sut.enable_direct_loadgen(True)

        batch_size = self.batch_size
        num_samples = len(query_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size

        print(f"[Offline] {num_samples} samples, batch_size={batch_size}", file=sys.stderr)

        # Phase 1: Preprocessing + Submit
        print(f"[Preprocess] ", end="", file=sys.stderr)
        i = 0
        batches_submitted = 0
        while i < num_samples:
            end_idx = min(i + batch_size, num_samples)
            actual_batch_size = end_idx - i

            query_ids = []
            sample_indices = []
            batch_data = []

            for j in range(i, end_idx):
                qs = query_samples[j]
                query_ids.append(qs.id)
                sample_indices.append(qs.index)

                input_data = self._get_input_data(qs.index)
                if input_data.ndim == 4 and input_data.shape[0] == 1:
                    input_data = input_data[0]
                batch_data.append(input_data)

            if actual_batch_size == batch_size:
                batched_input = np.stack(batch_data, axis=0)
            else:
                padded = batch_data + [batch_data[-1]] * (batch_size - actual_batch_size)
                batched_input = np.stack(padded, axis=0)

            if batched_input.dtype != np.float32:
                batched_input = batched_input.astype(np.float32)
            if not batched_input.flags['C_CONTIGUOUS']:
                batched_input = np.ascontiguousarray(batched_input)

            self._cpp_sut.start_async_batch(
                batched_input, query_ids, sample_indices, actual_batch_size
            )

            batches_submitted += 1
            # Progress bar: show dot every 10% or every batch if < 10
            if num_batches <= 10 or batches_submitted % max(1, num_batches // 10) == 0:
                print(".", end="", file=sys.stderr, flush=True)

            i = end_idx

        print(f" {batches_submitted}/{num_batches} batches", file=sys.stderr)

        # Phase 2: Wait for completion with progress
        print(f"[Inference] ", end="", file=sys.stderr)
        last_completed = 0
        wait_start = time.time()
        dots_printed = 0

        while True:
            completed = self._cpp_sut.get_completed_count()
            if completed >= num_samples:
                break

            # Print progress dot every 10%
            progress = int(completed * 10 / num_samples)
            while dots_printed < progress:
                print(".", end="", file=sys.stderr, flush=True)
                dots_printed += 1

            # Check for stall (no progress for 30s)
            if completed > last_completed:
                last_completed = completed
                wait_start = time.time()
            elif time.time() - wait_start > 30:
                print(f"\n[WARN] Stalled at {completed}/{num_samples}", file=sys.stderr)
                wait_start = time.time()

            time.sleep(0.1)

        # Final dots
        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(f" {num_samples}/{num_samples} ({elapsed:.1f}s, {num_samples/elapsed:.1f} qps)", file=sys.stderr)
        self._query_count += 1

    def _issue_query_server(self, query_samples: List) -> None:
        """Process queries in Server mode."""
        if not self._server_callback_set:
            self._cpp_sut.enable_direct_loadgen(True)
            self._server_callback_set = True

        if not self._qsl_validated:
            self._prevalidate_qsl_data()
            self._qsl_validated = True

        if self._cpp_qsl_registered:
            query_ids = [qs.id for qs in query_samples]
            sample_indices = [qs.index for qs in query_samples]
            self._cpp_sut.issue_queries_server_fast(query_ids, sample_indices)
        else:
            # Fallback path
            for qs in query_samples:
                input_data = self._get_input_data(qs.index)
                if input_data.ndim == 3:
                    input_data = input_data[np.newaxis, ...]
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                if not input_data.flags['C_CONTIGUOUS']:
                    input_data = np.ascontiguousarray(input_data)

                self._cpp_sut.start_async_batch(
                    input_data, [qs.id], [qs.index], 1
                )

        self._query_count += 1

    def issue_queries(self, query_samples: List) -> None:
        """Process incoming queries."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush pending queries."""
        self._cpp_sut.wait_all()

    def get_sut(self):
        """Get LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def supports_native_benchmark(self) -> bool:
        """Check if native C++ benchmark is supported."""
        return self.scenario == Scenario.SERVER

    def run_native_benchmark(
        self,
        test_settings: "lg.TestSettings",
        log_settings: "lg.LogSettings",
    ) -> None:
        """Run benchmark with pure C++ SUT."""
        if not self.supports_native_benchmark():
            raise RuntimeError("Native benchmark not supported for this scenario")

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # For accuracy mode, load ALL samples (LoadGen queries all of them)
        # For performance mode, only load performance_sample_count
        if self._is_accuracy_mode:
            num_samples = self.qsl.total_sample_count
        else:
            num_samples = min(self.qsl.performance_sample_count, self.qsl.total_sample_count)

        sample_indices = list(range(num_samples))
        self.qsl.load_query_samples(sample_indices)

        # Register samples in C++
        self._prevalidate_qsl_data()

        if not self._cpp_qsl_registered:
            raise RuntimeError("Failed to register samples in C++")

        total_count = self.qsl.total_sample_count
        perf_count = self.qsl.performance_sample_count

        target_qps = getattr(test_settings, 'server_target_qps', 0.0)
        target_latency_ns = getattr(test_settings, 'server_target_latency_ns', 0)
        min_duration_ms = getattr(test_settings, 'min_duration_ms', 0)
        min_query_count = getattr(test_settings, 'min_query_count', 0)

        self._cpp_sut.reset_counters()

        log_output_dir = getattr(log_settings.log_output, 'outdir', '.')

        self._cpp_sut.run_server_benchmark(
            total_count,
            perf_count,
            "",
            "",
            log_output_dir,
            target_qps,
            target_latency_ns,
            min_duration_ms,
            min_query_count,
            self._is_accuracy_mode
        )

        self._query_count = self._cpp_sut.get_issued_count()

        self.qsl.unload_query_samples(sample_indices)

    def get_qsl(self):
        """Get LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"RetinaNetMultiDieCpp-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get stored predictions formatted for accuracy calculation."""
        cpp_preds = self._cpp_sut.get_predictions()
        result = {}
        for idx, pred in cpp_preds.items():
            # pred is a dict with 'boxes', 'scores', 'labels', 'num_detections'
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'] is not None else np.array([])

            # Reshape boxes to [N, 4] if needed
            if boxes.size > 0 and boxes.ndim == 1:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            }
        return result

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._cpp_sut.set_store_predictions(store)
        self._is_accuracy_mode = store  # Also update for native benchmark

    def reset(self) -> None:
        """Reset state."""
        self._cpp_sut.reset_counters()
        self._query_count = 0
        self._server_callback_set = False
        self._qsl_validated = False
        try:
            self._cpp_sut.clear_sample_data()
        except Exception:
            pass
        try:
            self._cpp_sut.enable_direct_loadgen(False)
        except Exception:
            pass
        self._cpp_qsl_registered = False
        self._qsl_data_ready = False

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute mAP accuracy using pycocotools (official MLPerf method).

        Returns:
            Dictionary with mAP and other metrics
        """
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions (sample_count={self._sample_count}, query_count={self._query_count})")
            return {'mAP': 0.0, 'num_samples': 0}

        # Try to use pycocotools for accurate mAP calculation
        try:
            from ..datasets.coco_eval import evaluate_openimages_accuracy, PYCOCOTOOLS_AVAILABLE

            if PYCOCOTOOLS_AVAILABLE:
                # Find COCO annotations file
                coco_file = None
                data_path = Path(self.qsl.dataset.data_path)

                for name in [
                    "annotations/openimages-mlperf.json",
                    "openimages-mlperf.json",
                ]:
                    path = data_path / name
                    if path.exists():
                        coco_file = str(path)
                        break

                if coco_file:
                    logger.info(f"Using pycocotools with {coco_file}")

                    # Get sample_idx to filename mapping for correct COCO image_id lookup
                    sample_to_filename = None
                    if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                        sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                        logger.info(f"Got filename mapping for {len(sample_to_filename)} samples")

                    return evaluate_openimages_accuracy(
                        predictions=predictions,
                        coco_annotations_file=coco_file,
                        input_size=800,
                        model_labels_zero_indexed=True,
                        boxes_in_pixels=True,
                        sample_to_filename=sample_to_filename,
                    )
                else:
                    logger.warning("COCO annotations file not found, using fallback mAP calculation")
        except Exception as e:
            logger.warning(f"pycocotools evaluation failed: {e}, using fallback")
            import traceback
            traceback.print_exc()

        # Fallback to basic mAP calculation
        pred_list = []
        indices = []

        for sample_idx in sorted(predictions.keys()):
            pred_list.append(predictions[sample_idx])
            indices.append(sample_idx)

        return self.qsl.dataset.compute_accuracy(pred_list, indices)


def is_retinanet_multi_die_cpp_available() -> bool:
    """Check if C++ multi-die SUT is available."""
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
