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

try:
    from ..cpp import (
        ResNetCppSUT, BertCppSUT, RetinaNetCppSUT, SSDResNet34CppSUT,
        CPP_AVAILABLE,
    )
except ImportError:
    CPP_AVAILABLE = False
    ResNetCppSUT = None
    BertCppSUT = None
    RetinaNetCppSUT = None
    SSDResNet34CppSUT = None

logger = logging.getLogger(__name__)


class ProgressMonitor:

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
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._print_progress(final=True)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.update_interval)

    def _print_progress(self, final: bool = False):
        completed = self.get_completed()
        total = self.total_samples
        elapsed = time.time() - self._start_time

        if total > 0:
            pct = min(100.0, completed / total * 100)
            bar_width = 40
            filled = int(bar_width * completed / total)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

            throughput = completed / elapsed if elapsed > 0 else 0

            status = f"\r{self.name}: [{bar}] {completed}/{total} ({pct:.1f}%) | {throughput:.1f} samples/s"

            if final:
                status += f" | Total: {elapsed:.1f}s\n"
            else:
                status += "   "  # Extra spaces to clear previous longer output

            sys.stderr.write(status)
            sys.stderr.flush()


class ResNetCppSUTWrapper:

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
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

        device = config.openvino.device
        num_streams = 0  # Auto
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        # Determine if NHWC input is needed (default is NHWC)
        use_nhwc = True
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

        self._cpp_sut = ResNetCppSUT(model_path, device, num_streams, performance_hint, use_nhwc)
        self._cpp_sut.load()

        self.input_name = self._cpp_sut.get_input_name()
        self.output_name = self._cpp_sut.get_output_name()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        self._setup_response_callback()

        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        def response_callback(query_id: int, output_data):
            # Called from C++ thread; C++ binding handles GIL acquisition
            try:
                if output_data is not None:
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
                # Still try to complete to avoid LoadGen hang
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
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

            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)

            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            # GIL released during C++ inference
            cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"OpenVINO-Cpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Any]:
        # C++ returns Dict[int, List[float]], convert to numpy arrays
        raw_predictions = self._cpp_sut.get_predictions()
        return {idx: np.array(pred, dtype=np.float32) for idx, pred in raw_predictions.items()}

    def reset(self) -> None:
        self._cpp_sut.reset_counters()


class BertCppSUTWrapper:

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

        self.input_ids_name = self._cpp_sut.get_input_ids_name()
        self.attention_mask_name = self._cpp_sut.get_attention_mask_name()
        self.token_type_ids_name = self._cpp_sut.get_token_type_ids_name()
        self.start_logits_name = self._cpp_sut.get_start_logits_name()
        self.end_logits_name = self._cpp_sut.get_end_logits_name()
        self.seq_length = self._cpp_sut.get_seq_length()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        self._setup_response_callback()

        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        import array

        dummy_data = array.array('B', [0] * 8)
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, start_logits, end_logits):
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List[Any]) -> None:
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
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
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
        raw_preds = self._cpp_sut.get_predictions()
        # Convert from dict with 'start_logits' and 'end_logits' keys to tuple
        result = {}
        for idx, pred in raw_preds.items():
            start_logits = np.array(pred['start_logits'], dtype=np.float32)
            end_logits = np.array(pred['end_logits'], dtype=np.float32)
            result[idx] = (start_logits, end_logits)
        return result

    def reset(self) -> None:
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        indices = sorted(predictions.keys())

        pred_texts = self.qsl.dataset.postprocess(
            [(predictions[idx][0], predictions[idx][1]) for idx in indices],
            indices
        )

        return self.qsl.dataset.compute_accuracy(pred_texts, indices)


def create_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: QuerySampleLibrary,
    scenario: Scenario = Scenario.SERVER,
    force_python: bool = False,
):
    # Accelerator devices are not supported by C++ SUT - they require MultiDeviceSUT
    if config.openvino.is_accelerator_device():
        device = config.openvino.device
        raise ValueError(
            f"C++ SUT does not support accelerator devices (got: {device}). "
            "Use MultiDeviceSUT via BenchmarkRunner._create_sut_for_backend() instead."
        )

    if CPP_AVAILABLE and not force_python:
        logger.info(f"Using ResNet C++ SUT on {config.openvino.device}")
        return ResNetCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python SUT
    logger.info(f"Using ResNet Python SUT on {config.openvino.device}")
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
    # Accelerator devices are not supported by C++ SUT - they require MultiDeviceSUT
    if config.openvino.is_accelerator_device():
        device = config.openvino.device
        raise ValueError(
            f"C++ SUT does not support accelerator devices (got: {device}). "
            "Use MultiDeviceSUT via BenchmarkRunner._create_sut_for_backend() instead."
        )

    if CPP_AVAILABLE and BertCppSUT is not None and not force_python:
        logger.info(f"Using BERT C++ SUT on {config.openvino.device}")
        return BertCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python BertSUT
    logger.info(f"Using BERT Python SUT on {config.openvino.device}")
    from .bert_sut import BertSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return BertSUT(config, backend, qsl, scenario)


class RetinaNetCppSUTWrapper:

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

        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        # Determine if NHWC input is needed (default is NHWC)
        use_nhwc = True
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

        self._cpp_sut = RetinaNetCppSUT(model_path, device, num_streams, performance_hint, use_nhwc)
        self._cpp_sut.load()

        self.input_name = self._cpp_sut.get_input_name()
        self.boxes_name = self._cpp_sut.get_boxes_name()
        self.scores_name = self._cpp_sut.get_scores_name()
        self.labels_name = self._cpp_sut.get_labels_name()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        self._setup_response_callback()

        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        import array

        dummy_data = array.array('B', [0] * 8)
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, boxes, scores, labels):
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List[Any]) -> None:
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
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
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
        """Note: MLPerf RetinaNet ONNX model outputs boxes in PIXEL coordinates [0, 800]
        as [x1, y1, x2, y2], scores as [N] confidence, labels as ORIGINAL OpenImages
        category IDs (not 0-indexed)."""
        raw_preds = self._cpp_sut.get_predictions()
        result = {}

        for idx, pred in raw_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'].size > 0 else np.array([])

            # Reshape boxes to [N, 4]; keep pixel coordinates as-is for coco_eval
            if len(boxes) > 0 and len(boxes) % 4 == 0:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels.astype(np.int64) if len(labels) > 0 else np.array([], dtype=np.int64),
            }
        return result

    def reset(self) -> None:
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions collected (issued={self._cpp_sut.get_issued_count()}, "
                          f"completed={self._cpp_sut.get_completed_count()})")
            return {'mAP': 0.0, 'num_samples': 0}

        try:
            from ..datasets.coco_eval import evaluate_openimages_accuracy, PYCOCOTOOLS_AVAILABLE

            if PYCOCOTOOLS_AVAILABLE:
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

                    # CRITICAL: dataset order may differ from COCO annotation order
                    sample_to_filename = None
                    if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                        sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                        logger.debug(f"Got filename mapping for {len(sample_to_filename)} samples")

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
    # Accelerator devices are not supported by C++ SUT - they require MultiDeviceSUT
    if config.openvino.is_accelerator_device():
        device = config.openvino.device
        raise ValueError(
            f"C++ SUT does not support accelerator devices (got: {device}). "
            "Use MultiDeviceSUT via BenchmarkRunner._create_sut_for_backend() instead."
        )

    if CPP_AVAILABLE and RetinaNetCppSUT is not None and not force_python:
        logger.info(f"Using RetinaNet C++ SUT on {config.openvino.device}")
        return RetinaNetCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python RetinaNetSUT
    logger.info(f"Using RetinaNet Python SUT on {config.openvino.device}")
    from .retinanet_sut import RetinaNetSUT
    from ..backends.openvino_backend import OpenVINOBackend

    # Determine if NHWC input is needed (default is NHWC)
    use_nhwc = True
    if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
        use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

    backend = OpenVINOBackend(model_path, config.openvino, use_nhwc_input=use_nhwc)
    return RetinaNetSUT(config, backend, qsl, scenario)


# =============================================================================
# SSD-ResNet34 SUT Wrapper
# =============================================================================


class SSDResNet34CppSUTWrapper:

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: "QuerySampleLibrary",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not CPP_AVAILABLE or SSDResNet34CppSUT is None:
            raise ImportError(
                "SSD-ResNet34 C++ SUT extension not available. "
                "Build it with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        use_nhwc = True
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

        self._cpp_sut = SSDResNet34CppSUT(model_path, device, num_streams, performance_hint, use_nhwc)
        self._cpp_sut.load()

        self.input_name = self._cpp_sut.get_input_name()
        self.boxes_name = self._cpp_sut.get_boxes_name()
        self.scores_name = self._cpp_sut.get_scores_name()
        self.labels_name = self._cpp_sut.get_labels_name()

        self._cpp_sut.set_store_predictions(True)
        self._query_to_sample: Dict[int, int] = {}  # query_id -> sample_idx
        self._setup_response_callback()
        self._progress_monitor: Optional[ProgressMonitor] = None

    def _setup_response_callback(self):
        import array

        def response_callback(query_id: int, boxes, scores, labels):
            sample_idx = self._query_to_sample.pop(query_id, 0)

            if boxes is not None and scores is not None:
                boxes_arr = np.asarray(boxes, dtype=np.float32)
                scores_arr = np.asarray(scores, dtype=np.float32)
                labels_arr = np.asarray(labels, dtype=np.float32) if labels is not None else None

                if len(boxes_arr) > 0 and len(boxes_arr) % 4 == 0:
                    boxes_arr = boxes_arr.reshape(-1, 4)

                response_data = []
                for det_idx in range(len(scores_arr)):
                    score = float(scores_arr[det_idx])
                    if score < 0.5:
                        continue
                    box = boxes_arr[det_idx]
                    label = float(labels_arr[det_idx]) if labels_arr is not None and det_idx < len(labels_arr) else 0.0
                    response_data.extend([
                        float(sample_idx),
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                        score,
                        label,
                    ])

                if response_data:
                    data_np = np.array(response_data, dtype=np.float32)
                    response_array = array.array('B', data_np.tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
                else:
                    response = lg.QuerySampleResponse(query_id, 0, 0)
            else:
                response = lg.QuerySampleResponse(query_id, 0, 0)

            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def issue_queries(self, query_samples: List[Any]) -> None:
        if self._progress_monitor is None:
            self._progress_monitor = ProgressMonitor(
                total_samples=self.qsl.total_sample_count,
                get_completed=self._cpp_sut.get_completed_count,
                name="SSD-ResNet34",
            )
            self._progress_monitor.start()

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Store mapping for MLCommons accuracy log response formatting
            self._query_to_sample[query_id] = sample_idx

            features = self.qsl.get_features(sample_idx)
            input_data = features.get('input', features.get(self.input_name))
            input_data = np.ascontiguousarray(input_data.flatten(), dtype=np.float32)
            self._cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        self._cpp_sut.wait_all()
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor = None

    def get_sut(self) -> "lg.ConstructSUT":
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"OpenVINO-SSDResNet34Cpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        raw_preds = self._cpp_sut.get_predictions()
        result = {}

        for idx, pred in raw_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'].size > 0 else np.array([])

            if len(boxes) > 0 and len(boxes) % 4 == 0:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels.astype(np.int64) if len(labels) > 0 else np.array([], dtype=np.int64),
            }
        return result

    def reset(self) -> None:
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions collected (issued={self._cpp_sut.get_issued_count()}, "
                          f"completed={self._cpp_sut.get_completed_count()})")
            return {'mAP': 0.0, 'num_samples': 0}

        pred_list = []
        indices = sorted(predictions.keys())
        for idx in indices:
            pred_list.append(predictions[idx])

        return self.qsl.dataset.compute_accuracy(pred_list, indices)


def create_ssd_resnet34_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
    force_python: bool = False,
):
    if config.openvino.is_accelerator_device():
        device = config.openvino.device
        raise ValueError(
            f"C++ SUT does not support accelerator devices (got: {device}). "
            "Use MultiDeviceSUT via BenchmarkRunner._create_sut_for_backend() instead."
        )

    if CPP_AVAILABLE and SSDResNet34CppSUT is not None and not force_python:
        logger.info(f"Using SSD-ResNet34 C++ SUT on {config.openvino.device}")
        return SSDResNet34CppSUTWrapper(config, model_path, qsl, scenario)

    logger.info(f"Using SSD-ResNet34 Python SUT on {config.openvino.device}")
    from .ssd_resnet34_sut import SSDResNet34SUT
    from ..backends.openvino_backend import OpenVINOBackend

    use_nhwc = True
    if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
        use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

    backend = OpenVINOBackend(model_path, config.openvino, use_nhwc_input=use_nhwc)
    return SSDResNet34SUT(config, backend, qsl, scenario)


def create_unet3d_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
):
    """Create 3D UNET SUT for single-device (CPU)."""
    if config.openvino.is_accelerator_device():
        device = config.openvino.device
        raise ValueError(
            f"Single-device SUT does not support accelerator devices (got: {device}). "
            "Use SUTFactory.create_multi_die_sut() instead."
        )

    logger.info(f"Using 3D UNET Python SUT on {config.openvino.device}")
    from .unet3d_sut import UNet3DSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino, use_nhwc_input=False)
    backend.load()
    return UNet3DSUT(config, backend, qsl, scenario)
