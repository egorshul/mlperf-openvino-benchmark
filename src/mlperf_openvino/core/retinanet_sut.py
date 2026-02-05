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
from ..datasets.openimages import OpenImagesQSL

logger = logging.getLogger(__name__)

# RetinaNet constants for OpenImages (MLPerf uses 264 classes, not COCO's 80)
NUM_CLASSES = 264
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 100


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class RetinaNetSUT:

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: OpenImagesQSL,
        scenario: Scenario = Scenario.OFFLINE,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
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
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        if not self.backend.is_loaded:
            self.backend.load()

        self._map_io_names()

        self._predictions: Dict[int, Dict[str, np.ndarray]] = {}
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
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
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

        self.boxes_name = None
        self.scores_name = None
        self.labels_name = None

        for name in model_outputs:
            name_lower = name.lower()
            if 'box' in name_lower or 'bbox' in name_lower:
                self.boxes_name = name
            elif 'score' in name_lower or 'conf' in name_lower:
                self.scores_name = name
            elif 'label' in name_lower or 'class' in name_lower:
                self.labels_name = name

        # Fallback: assume positional output order [boxes, scores, labels]
        if not self.boxes_name and len(model_outputs) >= 1:
            self.boxes_name = model_outputs[0]
        if not self.scores_name and len(model_outputs) >= 2:
            self.scores_name = model_outputs[1]
        if not self.labels_name and len(model_outputs) >= 3:
            self.labels_name = model_outputs[2]

    def _postprocess_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        # Handle batch dimension in model outputs
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if labels is not None and labels.ndim == 2:
            labels = labels[0]

        # If scores are per-class (shape [N, NUM_CLASSES]), reduce to max
        if scores.ndim == 2 and scores.shape[1] == NUM_CLASSES:
            labels = np.argmax(scores, axis=1)
            scores = np.max(scores, axis=1)

        mask = scores > self.score_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        if labels is not None:
            filtered_labels = labels[mask]
        else:
            filtered_labels = np.zeros(len(filtered_boxes), dtype=np.int64)

        final_boxes = []
        final_scores = []
        final_labels = []

        for class_id in range(NUM_CLASSES):
            class_mask = filtered_labels == class_id
            if not class_mask.any():
                continue

            class_boxes = filtered_boxes[class_mask]
            class_scores = filtered_scores[class_mask]

            keep = nms(class_boxes, class_scores, self.nms_threshold)

            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(np.full(len(keep), class_id))

        if final_boxes:
            final_boxes = np.concatenate(final_boxes)
            final_scores = np.concatenate(final_scores)
            final_labels = np.concatenate(final_labels)

            order = np.argsort(final_scores)[::-1][:self.max_detections]
            final_boxes = final_boxes[order]
            final_scores = final_scores[order]
            final_labels = final_labels[order]
        else:
            final_boxes = np.array([])
            final_scores = np.array([])
            final_labels = np.array([])

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels,
        }

    def _process_sample(self, sample_idx: int) -> Dict[str, np.ndarray]:
        features = self.qsl.get_features(sample_idx)
        inputs = {self.input_name: features['input']}
        outputs = self.backend.predict(inputs)

        boxes = outputs.get(self.boxes_name, np.array([]))
        scores = outputs.get(self.scores_name, np.array([]))
        labels = outputs.get(self.labels_name) if self.labels_name else None

        return self._postprocess_detections(boxes, scores, labels)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="RetinaNet Offline inference")

        for qs in query_samples:
            sample_idx = qs.index
            detections = self._process_sample(sample_idx)
            self._predictions[sample_idx] = detections

            num_detections = len(detections['boxes'])
            response_data = np.array([num_detections], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                qs.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            self._sample_count += 1
            self._update_progress(1)

        self._close_progress()

        lg.QuerySamplesComplete(responses)

        self._query_count += 1

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete

        if self._query_count == 0:
            self._start_progress(0, desc="RetinaNet Server inference")

        for qs in query_samples:
            sample_idx = qs.index
            detections = self._process_sample(sample_idx)
            self._predictions[sample_idx] = detections

            num_detections = len(detections['boxes'])
            response_data = np.array([num_detections], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                qs.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            self._sample_count += 1
            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

        self._query_count += 1

    def issue_queries(self, query_samples: List[Any]) -> None:
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

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
        return f"RetinaNet-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        if not self._predictions:
            logger.warning(f"No predictions (sample_count={self._sample_count}, query_count={self._query_count})")
            return {'mAP': 0.0, 'num_samples': 0}

        try:
            from ..datasets.coco_eval import evaluate_openimages_accuracy, PYCOCOTOOLS_AVAILABLE
            from pathlib import Path

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
                    logger.info(f"Using pycocotools with {coco_file}")

                    # Dataset order may differ from COCO annotation order
                    sample_to_filename = None
                    if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                        sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                        logger.info(f"Got filename mapping for {len(sample_to_filename)} samples")

                    return evaluate_openimages_accuracy(
                        predictions=self._predictions,
                        coco_annotations_file=coco_file,
                        input_size=800,
                        model_labels_zero_indexed=True,  # Model outputs 0-indexed (0-263), COCO expects 1-indexed (1-264)
                        boxes_in_pixels=True,  # Model outputs pixel coords [0,800]
                        sample_to_filename=sample_to_filename,
                    )
                else:
                    logger.warning("COCO annotations file not found, using fallback mAP calculation")
        except Exception as e:
            logger.warning(f"pycocotools evaluation failed: {e}, using fallback")
            import traceback
            traceback.print_exc()

        predictions = []
        indices = []

        for sample_idx in sorted(self._predictions.keys()):
            predictions.append(self._predictions[sample_idx])
            indices.append(sample_idx)

        return self.qsl.dataset.compute_accuracy(predictions, indices)
