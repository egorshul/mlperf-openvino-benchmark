import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from ..cpp import SSDResNet34MultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and SSDResNet34MultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    SSDResNet34MultiDieCppSUT = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .image_multi_die_sut_base import ImageMultiDieSUTBase
from .config import Scenario
from ..datasets.coco import LABEL_TO_COCO_ID

logger = logging.getLogger(__name__)


class SSDResNet34MultiDieCppSUTWrapper(ImageMultiDieSUTBase):

    MODEL_NAME = "SSD-ResNet34"
    DEFAULT_OFFLINE_BATCH_SIZE = 1  # 1200x1200 inputs are very large (~16.5MB)
    DEFAULT_OFFLINE_NIREQ_MULTIPLIER = 2  # Large inputs
    DEFAULT_SERVER_NIREQ_MULTIPLIER = 2
    DEFAULT_EXPLICIT_BATCH_SIZE = 1  # Large inputs
    DEFAULT_BATCH_TIMEOUT_US = 1500  # Longer timeout for larger data
    BATCH_SERVER_ACCURACY = True  # Use Offline-style batch dispatch for Server accuracy

    def supports_native_benchmark(self) -> bool:
        # SSD-ResNet34 accuracy uses COCO QSL with 5K samples that can't be
        # preloaded into C++ (lazy LRU cache). Disable native path for accuracy
        # so it goes through Python SUT with batch dispatch (BATCH_SERVER_ACCURACY).
        return self.scenario == Scenario.SERVER and not self._is_accuracy_mode

    def _check_cpp_availability(self) -> None:
        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

    def _create_cpp_sut(
        self,
        model_path: str,
        device_prefix: str,
        batch_size: int,
        compile_props: Dict,
        use_nhwc: bool,
        nireq_multiplier: int
    ) -> Any:
        return SSDResNet34MultiDieCppSUT(
            model_path,
            device_prefix,
            batch_size,
            compile_props,
            use_nhwc,
            nireq_multiplier
        )

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        cpp_preds = self._cpp_sut.get_predictions()
        result = {}
        for idx, pred in cpp_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'] is not None else np.array([])

            if boxes.size > 0 and boxes.ndim == 1:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            }
        return result

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions (sample_count={self._sample_count}, query_count={self._query_count})")
            return {'mAP': 0.0, 'num_samples': 0}

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            logger.error("pycocotools not available, cannot compute mAP")
            # Fallback to dataset-level accuracy
            pred_list = []
            indices = []
            for sample_idx in sorted(predictions.keys()):
                pred_list.append(predictions[sample_idx])
                indices.append(sample_idx)
            return self.qsl.dataset.compute_accuracy(pred_list, indices)

        # Find COCO annotations file
        coco_file = getattr(self.qsl.dataset, '_coco_annotations_file', None)
        if not coco_file:
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
            logger.error("COCO annotations file not found, cannot compute mAP")
            pred_list = []
            indices = []
            for sample_idx in sorted(predictions.keys()):
                pred_list.append(predictions[sample_idx])
                indices.append(sample_idx)
            return self.qsl.dataset.compute_accuracy(pred_list, indices)

        logger.info(f"Using pycocotools with {coco_file}")

        try:
            coco_gt = COCO(coco_file)

            sample_to_image_id = self.qsl.get_sample_to_image_id_mapping()
            coco_results = []
            evaluated_image_ids = []

            for sample_idx, pred in predictions.items():
                image_id = sample_to_image_id.get(sample_idx)
                if image_id is None:
                    continue

                evaluated_image_ids.append(image_id)

                boxes = pred.get('boxes', np.array([]))
                scores = pred.get('scores', np.array([]))
                labels = pred.get('labels', np.array([]))

                if len(boxes) == 0:
                    continue

                img_info = coco_gt.imgs.get(image_id, {})
                img_width = img_info.get('width', 1)
                img_height = img_info.get('height', 1)

                for box, score, label in zip(boxes, scores, labels):
                    if float(score) < 0.05:
                        continue
                    y1_px = float(box[0]) * img_height
                    x1_px = float(box[1]) * img_width
                    y2_px = float(box[2]) * img_height
                    x2_px = float(box[3]) * img_width

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
                logger.error("No valid predictions to evaluate after label mapping")
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
            logger.error(f"pycocotools evaluation failed: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to dataset-level accuracy
            pred_list = []
            indices = []
            for sample_idx in sorted(predictions.keys()):
                pred_list.append(predictions[sample_idx])
                indices.append(sample_idx)
            return self.qsl.dataset.compute_accuracy(pred_list, indices)


def is_ssd_resnet34_multi_die_cpp_available() -> bool:
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
