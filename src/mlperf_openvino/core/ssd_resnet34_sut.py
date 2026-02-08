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
from ..datasets.coco import COCOQSL, LABEL_TO_COCO_ID

logger = logging.getLogger(__name__)

# SSD-ResNet34 constants for COCO (80 COCO categories + 1 background)
NUM_CLASSES = 81
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 200


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


class SSDResNet34SUT:

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: COCOQSL,
        scenario: Scenario = Scenario.OFFLINE,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        model_has_nms: bool = True,
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
        # The MLPerf ONNX model (ssd_resnet34_mAP_20.2.onnx) has NMS baked in.
        # When True, skip redundant per-class NMS and just pass through model
        # output (matching the reference PostProcessCocoOnnx behavior).
        self.model_has_nms = model_has_nms

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

        # Fallback: assume positional output order [boxes, labels, scores]
        if not self.boxes_name and len(model_outputs) >= 1:
            self.boxes_name = model_outputs[0]
        if not self.labels_name and len(model_outputs) >= 2:
            self.labels_name = model_outputs[1]
        if not self.scores_name and len(model_outputs) >= 3:
            self.scores_name = model_outputs[2]

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

        # Skip class 0 (background) -- start from class 1
        for class_id in range(1, NUM_CLASSES):
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

    def _passthrough_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Pass through model output for NMS-baked ONNX model.

        Matches the MLCommons reference PostProcessCocoOnnx which only strips
        the batch dimension and filters by score — no additional NMS.
        """
        # Remove batch dimension
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if labels is not None and labels.ndim == 2:
            labels = labels[0]

        # Filter padding (model NMS already filtered at score > 0.05)
        mask = scores > 0.0
        boxes = boxes[mask]
        scores = scores[mask]
        if labels is not None:
            labels = labels[mask]
        else:
            labels = np.zeros(len(boxes), dtype=np.int64)

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
        }

    def _process_sample(self, sample_idx: int) -> Dict[str, np.ndarray]:
        features = self.qsl.get_features(sample_idx)
        inputs = {self.input_name: features['input']}
        outputs = self.backend.predict(inputs)

        boxes = outputs.get(self.boxes_name, np.array([]))
        scores = outputs.get(self.scores_name, np.array([]))
        labels = outputs.get(self.labels_name) if self.labels_name else None

        if self.model_has_nms:
            return self._passthrough_detections(boxes, scores, labels)
        return self._postprocess_detections(boxes, scores, labels)

    def _format_response_data(self, sample_idx: int, detections: Dict[str, np.ndarray]) -> np.ndarray:
        """Format detections as MLCommons accuracy-coco.py response data.

        Each detection = 7 float32: [qsl_idx, ymin, xmin, ymax, xmax, score, label].
        Model boxes are [x1, y1, x2, y2] normalized, reordered to [y1, x1, y2, x2].
        Labels are raw model output (1-indexed, 1-80); accuracy-coco.py with
        --use-inv-map converts to COCO category IDs.
        """
        boxes = detections.get('boxes', np.array([]))
        scores = detections.get('scores', np.array([]))
        labels = detections.get('labels', np.array([]))

        if len(boxes) == 0 or len(scores) == 0:
            return np.array([], dtype=np.float32)

        if boxes.ndim == 1 and len(boxes) % 4 == 0:
            boxes = boxes.reshape(-1, 4)

        response_vals = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score <= 0.0:
                continue
            box = boxes[i]
            label = float(labels[i]) if i < len(labels) else 0.0
            response_vals.extend([
                float(sample_idx),
                float(box[1]),  # ymin
                float(box[0]),  # xmin
                float(box[3]),  # ymax
                float(box[2]),  # xmax
                score,
                label,
            ])

        return np.array(response_vals, dtype=np.float32) if response_vals else np.array([], dtype=np.float32)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="SSD-ResNet34 Offline inference")

        for qs in query_samples:
            sample_idx = qs.index
            detections = self._process_sample(sample_idx)
            self._predictions[sample_idx] = detections

            # Format MLCommons accuracy-coco.py response data
            response_data = self._format_response_data(sample_idx, detections)
            if len(response_data) > 0:
                response_array = array.array('B', response_data.tobytes())
                response_arrays.append(response_array)
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            else:
                response = lg.QuerySampleResponse(qs.id, 0, 0)

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
            self._start_progress(0, desc="SSD-ResNet34 Server inference")

        for qs in query_samples:
            sample_idx = qs.index
            detections = self._process_sample(sample_idx)
            self._predictions[sample_idx] = detections

            # Format MLCommons accuracy-coco.py response data
            response_data = self._format_response_data(sample_idx, detections)
            if len(response_data) > 0:
                response_array = array.array('B', response_data.tobytes())
                response_arrays.append(response_array)
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            else:
                response = lg.QuerySampleResponse(qs.id, 0, 0)

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
        return f"SSD-ResNet34-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mAP using pycocotools, aligned with MLCommons accuracy-coco.py."""
        if not self._predictions:
            logger.warning(f"No predictions (sample_count={self._sample_count}, query_count={self._query_count})")
            return {'mAP': 0.0, 'num_samples': 0}

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            logger.warning("pycocotools not available, using fallback mAP")
            return self._fallback_accuracy()

        # Find COCO annotations file
        coco_file = getattr(self.qsl.dataset, '_coco_annotations_file', None)
        if not coco_file:
            from pathlib import Path
            data_path = Path(self.qsl.dataset.data_path)
            for name in [
                "annotations/instances_val2017.json",
                "instances_val2017.json",
            ]:
                path = data_path / name
                if path.exists():
                    coco_file = str(path)
                    break

        if not coco_file:
            logger.warning("COCO annotations file not found, using fallback mAP calculation")
            return self._fallback_accuracy()

        logger.info(f"Using pycocotools with {coco_file}")

        try:
            coco_gt = COCO(coco_file)

            # Get sample → image_id mapping directly from QSL
            sample_to_image_id = self.qsl.get_sample_to_image_id_mapping()

            # Build COCO-format detections per MLCommons accuracy-coco.py:
            # - Boxes are normalized [0, 1], scale to original image dimensions
            # - Labels are 1-indexed (1-80), map via inv_map = [0] + getCatIds()
            coco_results = []
            evaluated_image_ids = []

            for sample_idx, pred in self._predictions.items():
                image_id = sample_to_image_id.get(sample_idx)
                if image_id is None:
                    continue

                evaluated_image_ids.append(image_id)

                boxes = pred.get('boxes', np.array([]))
                scores = pred.get('scores', np.array([]))
                labels = pred.get('labels', np.array([]))

                if len(boxes) == 0:
                    continue

                # Get original image dimensions for coordinate denormalization
                img_info = coco_gt.imgs.get(image_id, {})
                img_width = img_info.get('width', 1)
                img_height = img_info.get('height', 1)

                for box, score, label in zip(boxes, scores, labels):
                    # Boxes are normalized [0, 1] per MLCommons SSD-ResNet34 reference
                    x1_px = float(box[0]) * img_width
                    y1_px = float(box[1]) * img_height
                    x2_px = float(box[2]) * img_width
                    y2_px = float(box[3]) * img_height

                    # Map 1-indexed label (1-80) to COCO category ID
                    # Matches MLCommons: inv_map = [0] + cocoGt.getCatIds()
                    label_int = int(label)
                    if label_int < 1 or label_int > 80:
                        continue
                    category_id = LABEL_TO_COCO_ID[label_int]

                    coco_results.append({
                        'image_id': int(image_id),
                        'category_id': category_id,
                        'bbox': [x1_px, y1_px, x2_px - x1_px, y2_px - y1_px],
                        'score': float(score),
                    })

            if not coco_results:
                logger.error("No predictions to evaluate after label mapping!")
                return {'mAP': 0.0, 'num_predictions': 0}

            logger.info(f"Evaluating {len(coco_results)} predictions on {len(evaluated_image_ids)} images")

            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            # Only evaluate on images that were actually processed
            coco_eval.params.imgIds = evaluated_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            results = {
                'mAP': float(coco_eval.stats[0]),
                'mAP@0.5': float(coco_eval.stats[1]),
                'mAP@0.75': float(coco_eval.stats[2]),
                'mAP_small': float(coco_eval.stats[3]),
                'mAP_medium': float(coco_eval.stats[4]),
                'mAP_large': float(coco_eval.stats[5]),
                'num_predictions': len(coco_results),
                'num_images': len(evaluated_image_ids),
            }

            logger.info(f"mAP@0.5:0.95 = {results['mAP']:.4f}")
            logger.info(f"mAP@0.5 = {results['mAP@0.5']:.4f}")

            return results

        except Exception as e:
            logger.warning(f"pycocotools evaluation failed: {e}, using fallback")
            import traceback
            traceback.print_exc()

        return self._fallback_accuracy()

    def _fallback_accuracy(self) -> Dict[str, float]:
        predictions = []
        indices = []

        for sample_idx in sorted(self._predictions.keys()):
            predictions.append(self._predictions[sample_idx])
            indices.append(sample_idx)

        return self.qsl.dataset.compute_accuracy(predictions, indices)
