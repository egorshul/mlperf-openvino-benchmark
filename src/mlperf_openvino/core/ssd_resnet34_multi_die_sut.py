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

logger = logging.getLogger(__name__)

# SSD-ResNet34 outputs 0-indexed labels (0-79) that must be mapped to
# non-contiguous COCO 2017 category IDs for evaluation with pycocotools.
LABEL_TO_COCO_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

INPUT_SIZE = 1200


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
            logger.error("pycocotools not available, cannot compute accurate mAP")
            return {'mAP': 0.0, 'error': 'pycocotools not installed'}

        coco_file = None
        if hasattr(self.qsl, 'dataset') and hasattr(self.qsl.dataset, '_coco_annotations_file'):
            coco_file = self.qsl.dataset._coco_annotations_file

        if not coco_file:
            # Fallback: search common annotation file locations
            if hasattr(self.qsl, 'dataset') and hasattr(self.qsl.dataset, 'data_path'):
                data_path = Path(self.qsl.dataset.data_path)
                for name in [
                    "annotations/instances_val2017.json",
                    "instances_val2017.json",
                    "annotations/coco-1200.json",
                    "coco-1200.json",
                ]:
                    path = data_path / name
                    if path.exists():
                        coco_file = str(path)
                        break

        if not coco_file:
            logger.error("COCO annotations file not found, cannot compute mAP")

            # Fall back to dataset compute_accuracy if available
            pred_list = []
            indices = []
            for sample_idx in sorted(predictions.keys()):
                pred_list.append(predictions[sample_idx])
                indices.append(sample_idx)
            return self.qsl.dataset.compute_accuracy(pred_list, indices)

        logger.info(f"Using pycocotools with {coco_file}")

        try:
            coco_gt = COCO(coco_file)

            # Build sample index to COCO image_id mapping
            sample_to_image_id = {}

            if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                logger.info(f"Got filename mapping for {len(sample_to_filename)} samples")

                filename_to_image_id = {}
                for img_id, img_info in coco_gt.imgs.items():
                    filename = img_info.get('file_name', '')
                    base_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                    filename_to_image_id[base_name] = img_id
                    filename_to_image_id[filename] = img_id

                for sample_idx, filename in sample_to_filename.items():
                    base_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                    if base_name in filename_to_image_id:
                        sample_to_image_id[sample_idx] = filename_to_image_id[base_name]
                    elif filename in filename_to_image_id:
                        sample_to_image_id[sample_idx] = filename_to_image_id[filename]
            else:
                # Fallback: assume sample order matches sorted COCO image_id order
                logger.warning("No filename mapping - assuming sample order matches COCO order")
                sorted_img_ids = sorted(coco_gt.imgs.keys())
                for sample_idx, img_id in enumerate(sorted_img_ids):
                    sample_to_image_id[sample_idx] = img_id

            # Convert predictions to COCO evaluation format
            coco_results = []
            skipped = 0

            for sample_idx, pred in predictions.items():
                boxes = pred.get('boxes', np.array([]))
                scores = pred.get('scores', np.array([]))
                labels = pred.get('labels', np.array([]))

                if len(boxes) == 0:
                    continue

                if sample_idx not in sample_to_image_id:
                    skipped += 1
                    continue
                image_id = sample_to_image_id[sample_idx]

                # Get original image dimensions for coordinate scaling
                if image_id in coco_gt.imgs:
                    img_info = coco_gt.imgs[image_id]
                    img_width = img_info.get('width', INPUT_SIZE)
                    img_height = img_info.get('height', INPUT_SIZE)
                else:
                    img_width = img_height = INPUT_SIZE

                for box, score, label in zip(boxes, scores, labels):
                    # Boxes are in model input pixel coordinates [0, INPUT_SIZE]
                    # Scale to original image dimensions
                    x1, y1, x2, y2 = box
                    scale_x = img_width / INPUT_SIZE
                    scale_y = img_height / INPUT_SIZE
                    x1_px = x1 * scale_x
                    y1_px = y1 * scale_y
                    x2_px = x2 * scale_x
                    y2_px = y2 * scale_y

                    # Convert to COCO format [x, y, w, h]
                    bbox_width = x2_px - x1_px
                    bbox_height = y2_px - y1_px

                    # Map model label (0-indexed) to COCO category_id
                    label_idx = int(label)
                    if 0 <= label_idx < len(LABEL_TO_COCO_ID):
                        category_id = LABEL_TO_COCO_ID[label_idx]
                    else:
                        # Label out of range, skip
                        continue

                    coco_results.append({
                        'image_id': int(image_id),
                        'category_id': category_id,
                        'bbox': [float(x1_px), float(y1_px), float(bbox_width), float(bbox_height)],
                        'score': float(score),
                    })

            if skipped > 0:
                logger.warning(f"Skipped {skipped} samples without image_id mapping")

            if not coco_results:
                logger.error("No valid predictions to evaluate after label mapping")
                return {'mAP': 0.0, 'num_predictions': 0}

            logger.info(f"Evaluating {len(coco_results)} predictions")

            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
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
