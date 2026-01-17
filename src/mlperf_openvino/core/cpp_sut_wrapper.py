"""
Wrapper for C++ SUT that integrates with the MLPerf LoadGen interface.

This module provides a Python wrapper around the C++ SUT extension,
handling the interface between LoadGen callbacks and C++ inference.
The actual inference runs in C++ without GIL for maximum throughput.
"""

import logging
import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

# Try to import C++ extension
try:
    from ..cpp import (
        ResNetCppSUT, BertCppSUT, RetinaNetCppSUT,
        CPP_AVAILABLE,
    )
except ImportError:
    CPP_AVAILABLE = False
    ResNetCppSUT = None
    BertCppSUT = None
    RetinaNetCppSUT = None

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """Progress bar monitor for SUT inference."""

    def __init__(
        self,
        total_samples: int,
        get_completed: Callable[[], int],
        name: str = "Progress",
        update_interval: float = 0.5,
    ):
        self.total_samples = total_samples
        self.get_completed = get_completed
        self.name = name
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self):
        """Start progress monitoring thread."""
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop progress monitoring and print final status."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._print_progress(final=True)

    def _monitor_loop(self):
        """Background thread that updates progress bar."""
        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.update_interval)

    def _print_progress(self, final: bool = False):
        """Print progress bar to stderr."""
        completed = self.get_completed()
        total = self.total_samples
        elapsed = time.time() - self._start_time

        if total > 0:
            pct = min(100.0, completed / total * 100)
            bar_width = 40
            filled = int(bar_width * completed / total)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Calculate throughput
            throughput = completed / elapsed if elapsed > 0 else 0

            status = f"\r{self.name}: [{bar}] {completed}/{total} ({pct:.1f}%) | {throughput:.1f} samples/s"

            if final:
                status += f" | Total: {elapsed:.1f}s\n"
            else:
                status += "   "  # Extra spaces to clear previous longer output

            sys.stderr.write(status)
            sys.stderr.flush()


class ResNetCppSUTWrapper:
    """
    Wrapper for ResNet C++ SUT that provides MLPerf LoadGen interface.

    This class handles:
    - LoadGen integration (issue_queries, flush_queries)
    - QSL sample loading and caching
    - Response callback from C++ to LoadGen
    - Statistics and progress monitoring

    The actual inference runs in C++ without Python GIL for maximum throughput.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize the ResNet C++ SUT wrapper.

        Args:
            config: Benchmark configuration
            model_path: Path to ONNX or OpenVINO IR model
            qsl: Query Sample Library
            scenario: Test scenario (SERVER or OFFLINE)
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ SUT extension is not available. "
                "Build it with: python -m mlperf_openvino.cpp.setup_cpp"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create C++ SUT
        device = config.openvino.device
        num_streams = 0  # Auto
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        self._cpp_sut = ResNetCppSUT(model_path, device, num_streams, performance_hint)
        self._cpp_sut.load()

        # Get input/output names
        self.input_name = self._cpp_sut.get_input_name()
        self.output_name = self._cpp_sut.get_output_name()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Set up response callback
        self._setup_response_callback()

        # Progress monitor (started when queries are issued)
        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""

        def response_callback(query_id: int, output_data):
            """
            Called from C++ when inference completes.

            This runs in C++ thread but needs GIL for LoadGen call.
            The C++ binding handles GIL acquisition.
            """
            try:
                if output_data is not None:
                    # Create LoadGen response with output data
                    response = lg.QuerySampleResponse(
                        query_id,
                        output_data.ctypes.data,
                        output_data.nbytes
                    )
                else:
                    # Error case - empty response
                    response = lg.QuerySampleResponse(query_id, 0, 0)

                lg.QuerySamplesComplete([response])
            except Exception as e:
                logger.error(f"Response callback error: {e}")
                # Still try to complete to avoid hang
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process incoming queries using C++ SUT.

        This method is called by LoadGen when queries need to be processed.
        It prepares the input data and calls C++ start_async for each sample.

        Args:
            query_samples: List of query samples from LoadGen
        """
        # Start progress monitor on first batch
        if self._progress_monitor is None:
            self._progress_monitor = ProgressMonitor(
                total_samples=self.qsl.total_sample_count,
                get_completed=self._cpp_sut.get_completed_count,
                name="ResNet",
            )
            self._progress_monitor.start()

        qsl = self.qsl
        cpp_sut = self._cpp_sut
        input_name = self.input_name

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data from QSL
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Ensure float32 for C++ SUT
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)

            # Ensure contiguous array
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            # Start async inference in C++ (GIL released during inference!)
            cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        """
        Flush any pending queries.

        This is called by LoadGen when all queries have been issued.
        """
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        """Get the LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        """Get the LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        """Get SUT name."""
        return f"OpenVINO-Cpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        """Get number of completed samples (for compatibility with benchmark_runner)."""
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        """Get number of issued queries (for compatibility with benchmark_runner)."""
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Any]:
        """Get all predictions (for accuracy mode)."""
        # C++ returns Dict[int, List[float]], convert to numpy arrays
        raw_predictions = self._cpp_sut.get_predictions()
        return {idx: np.array(pred, dtype=np.float32) for idx, pred in raw_predictions.items()}

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()


class BertCppSUTWrapper:
    """
    Wrapper for BERT C++ SUT with 3 int64 inputs and 2 float32 outputs.

    Optimized for BERT Question Answering:
    - input_ids, attention_mask, token_type_ids (int64)
    - start_logits, end_logits (float32)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: "QuerySampleLibrary",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not CPP_AVAILABLE or BertCppSUT is None:
            raise ImportError(
                "BERT C++ SUT extension not available. "
                "Build it with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create BERT C++ SUT
        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        self._cpp_sut = BertCppSUT(model_path, device, num_streams, performance_hint)
        self._cpp_sut.load()

        # Get input/output names
        self.input_ids_name = self._cpp_sut.get_input_ids_name()
        self.attention_mask_name = self._cpp_sut.get_attention_mask_name()
        self.token_type_ids_name = self._cpp_sut.get_token_type_ids_name()
        self.start_logits_name = self._cpp_sut.get_start_logits_name()
        self.end_logits_name = self._cpp_sut.get_end_logits_name()
        self.seq_length = self._cpp_sut.get_seq_length()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Set up response callback
        self._setup_response_callback()

        # Progress monitor (started when queries are issued)
        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""
        import array

        dummy_data = array.array('B', [0] * 8)
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, start_logits, end_logits):
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries using BERT C++ SUT."""
        if self._progress_monitor is None:
            self._progress_monitor = ProgressMonitor(
                total_samples=self.qsl.total_sample_count,
                get_completed=self._cpp_sut.get_completed_count,
                name="BERT",
            )
            self._progress_monitor.start()

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            features = self.qsl.get_features(sample_idx)
            self._cpp_sut.start_async(
                features['input_ids'],
                features['attention_mask'],
                features['token_type_ids'],
                query_id,
                sample_idx
            )

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        """Get the LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        """Get the LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"OpenVINO-BertCpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, tuple]:
        """Get all predictions as dict of {sample_idx: (start_logits, end_logits)}."""
        raw_preds = self._cpp_sut.get_predictions()
        # Convert from dict with 'start_logits' and 'end_logits' keys to tuple
        result = {}
        for idx, pred in raw_preds.items():
            start_logits = np.array(pred['start_logits'], dtype=np.float32)
            end_logits = np.array(pred['end_logits'], dtype=np.float32)
            result[idx] = (start_logits, end_logits)
        return result

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute BERT accuracy metrics (F1 and Exact Match).

        Returns:
            Dictionary with f1, exact_match, and num_samples
        """
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        # Extract indices in sorted order
        indices = sorted(predictions.keys())

        # Use dataset's postprocess to get answer texts
        pred_texts = self.qsl.dataset.postprocess(
            [(predictions[idx][0], predictions[idx][1]) for idx in indices],
            indices
        )

        # Compute F1 and EM using dataset's method
        return self.qsl.dataset.compute_accuracy(pred_texts, indices)


def create_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: QuerySampleLibrary,
    scenario: Scenario = Scenario.SERVER,
    force_python: bool = False,
):
    """
    Factory function to create the best available SUT.

    Uses async C++ SUT with InferRequest pool for maximum throughput.

    Args:
        config: Benchmark configuration
        model_path: Path to model file
        qsl: Query Sample Library
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        SUT instance
    """
    if CPP_AVAILABLE and not force_python:
        logger.info(f"Using ResNet C++ SUT for {scenario.value} mode")
        return ResNetCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python SUT
    logger.info(f"Using Python SUT for {scenario.value} mode")
    from .sut import OpenVINOSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return OpenVINOSUT(config, backend, qsl, scenario)


def create_bert_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
    force_python: bool = False,
):
    """
    Factory function to create BERT SUT.

    Uses C++ SUT if available for maximum performance.

    Args:
        config: Benchmark configuration
        model_path: Path to BERT model file
        qsl: Query Sample Library (SQuADQSL)
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        BERT SUT instance
    """
    if CPP_AVAILABLE and BertCppSUT is not None and not force_python:
        logger.info(f"Using BERT C++ SUT for {scenario.value} mode")
        return BertCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python BertSUT
    logger.info(f"Using Python BERT SUT for {scenario.value} mode")
    from .bert_sut import BertSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return BertSUT(config, backend, qsl, scenario)


class RetinaNetCppSUTWrapper:
    """
    Wrapper for RetinaNet C++ SUT with object detection outputs.

    Optimized for RetinaNet Object Detection:
    - 1 input: float32 image
    - 3 outputs: boxes, scores, labels
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: "QuerySampleLibrary",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not CPP_AVAILABLE or RetinaNetCppSUT is None:
            raise ImportError(
                "RetinaNet C++ SUT extension not available. "
                "Build it with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create RetinaNet C++ SUT
        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        self._cpp_sut = RetinaNetCppSUT(model_path, device, num_streams, performance_hint)
        self._cpp_sut.load()

        # Get info
        self.input_name = self._cpp_sut.get_input_name()
        self.boxes_name = self._cpp_sut.get_boxes_name()
        self.scores_name = self._cpp_sut.get_scores_name()
        self.labels_name = self._cpp_sut.get_labels_name()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Set up response callback
        self._setup_response_callback()

        # Progress monitor (started when queries are issued)
        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""
        import array

        dummy_data = array.array('B', [0] * 8)
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, boxes, scores, labels):
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries using RetinaNet C++ SUT."""
        if self._progress_monitor is None:
            self._progress_monitor = ProgressMonitor(
                total_samples=self.qsl.total_sample_count,
                get_completed=self._cpp_sut.get_completed_count,
                name="RetinaNet",
            )
            self._progress_monitor.start()

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            features = self.qsl.get_features(sample_idx)
            input_data = features.get('input', features.get(self.input_name))
            input_data = np.ascontiguousarray(input_data.flatten(), dtype=np.float32)
            self._cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        """Get the LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        """Get the LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"OpenVINO-RetinaNetCpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get all predictions as dict of {sample_idx: {boxes, scores, labels}}.

        Note: MLPerf RetinaNet ONNX model outputs:
        - boxes: [N, 4] in PIXEL coordinates [0, 800], format [x1, y1, x2, y2]
        - scores: [N] confidence scores
        - labels: [N] ORIGINAL OpenImages category IDs (not 0-indexed!)
        """
        raw_preds = self._cpp_sut.get_predictions()
        result = {}

        for idx, pred in raw_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'].size > 0 else np.array([])

            # Reshape boxes to [N, 4]
            if len(boxes) > 0 and len(boxes) % 4 == 0:
                boxes = boxes.reshape(-1, 4)
                # Model outputs boxes in PIXEL coordinates [0, 800]
                # Keep as-is, coco_eval expects pixel coordinates

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels.astype(np.int64) if len(labels) > 0 else np.array([], dtype=np.int64),
            }
        return result

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mAP accuracy using pycocotools (official MLPerf method)."""
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions collected (issued={self._cpp_sut.get_issued_count()}, "
                          f"completed={self._cpp_sut.get_completed_count()})")
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
                    logger.debug(f"Using pycocotools with {coco_file}")

                    # Get sample_idx to filename mapping for correct COCO image_id lookup
                    # This is CRITICAL - dataset order may differ from COCO annotation order!
                    sample_to_filename = None
                    if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                        sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                        logger.debug(f"Got filename mapping for {len(sample_to_filename)} samples")

                    return evaluate_openimages_accuracy(
                        predictions=predictions,
                        coco_annotations_file=coco_file,
                        input_size=800,
                        model_labels_zero_indexed=True,  # Model outputs 0-indexed labels (0-263), add +1 for category_ids (1-264)
                        boxes_in_pixels=True,  # Model outputs boxes in pixel coords [0,800]
                        sample_to_filename=sample_to_filename,
                    )
                else:
                    logger.warning("COCO annotations file not found, using fallback mAP calculation")
        except Exception as e:
            logger.warning(f"pycocotools evaluation failed: {e}, using fallback")
            import traceback
            traceback.print_exc()

        # Fallback to our mAP calculation
        pred_list = []
        indices = sorted(predictions.keys())

        for idx in indices:
            pred_list.append(predictions[idx])

        return self.qsl.dataset.compute_accuracy(pred_list, indices)


def create_retinanet_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
    force_python: bool = False,
):
    """
    Factory function to create RetinaNet SUT.

    Uses C++ SUT if available for maximum performance.

    Args:
        config: Benchmark configuration
        model_path: Path to RetinaNet model file
        qsl: Query Sample Library (OpenImagesQSL)
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        RetinaNet SUT instance
    """
    if CPP_AVAILABLE and RetinaNetCppSUT is not None and not force_python:
        logger.info(f"Using RetinaNet C++ SUT for {scenario.value} mode")
        return RetinaNetCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python RetinaNetSUT
    logger.info(f"Using Python RetinaNet SUT for {scenario.value} mode")
    from .retinanet_sut import RetinaNetSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return RetinaNetSUT(config, backend, qsl, scenario)
