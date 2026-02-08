"""COCO 2017 dataset for SSD-ResNet34 Object Detection."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# SSD-ResNet34 preprocessing constants
INPUT_SIZE = 1200
# ImageNet normalization (applied after /255.0)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# COCO 2017 has 80 categories
NUM_CLASSES = 80

# Official COCO 2017 category IDs (not contiguous: 1-90 with gaps)
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

# MLCommons-compatible inverse label map: inv_map = [0] + cocoGt.getCatIds()
# Index 0 = background (unused by NMS-baked model)
# Indices 1-80 = COCO category IDs for the 80 classes
# Model outputs 1-indexed labels (1-80), so: category_id = LABEL_TO_COCO_ID[label]
LABEL_TO_COCO_ID = [0] + list(COCO_CATEGORY_IDS)  # 81 elements, indices 0-80

# Reverse mapping: COCO category ID -> contiguous 1-indexed label
COCO_ID_TO_LABEL = {cid: i + 1 for i, cid in enumerate(COCO_CATEGORY_IDS)}


class COCODataset(BaseDataset):
    """COCO 2017 validation dataset for SSD-ResNet34."""

    def __init__(
        self,
        data_path: str,
        annotations_file: Optional[str] = None,
        count: Optional[int] = None,
        input_size: int = INPUT_SIZE,
        cache_preprocessed: bool = True,
        use_disk_cache: bool = True,
        output_layout: str = "NHWC",
        use_opencv: bool = True,
    ):
        if not PIL_AVAILABLE and not CV2_AVAILABLE:
            raise ImportError("Either Pillow or OpenCV is required")

        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.annotations_file = annotations_file
        self.input_size = input_size
        self.cache_preprocessed = cache_preprocessed
        self.use_disk_cache = use_disk_cache
        self.output_layout = output_layout
        self.use_opencv = use_opencv and CV2_AVAILABLE

        self._samples: List[Dict[str, Any]] = []
        self._annotations: Dict[int, List[Dict]] = defaultdict(list)
        self._image_info: Dict[int, Dict] = {}
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_lock = None
        self._is_loaded = False
        self._preprocessed_dir: Optional[Path] = None

        # For pycocotools evaluation
        self._coco_annotations_file: Optional[str] = None

    def load(self) -> None:
        if self._is_loaded:
            return

        logger.info(f"Loading COCO 2017 dataset from {self.data_path}")

        # Find images directory
        images_dir = self.data_path / "val2017"
        if not images_dir.exists():
            images_dir = self.data_path / "images"
        if not images_dir.exists():
            images_dir = self.data_path

        # Find annotations
        annotations_path = None
        if self.annotations_file:
            annotations_path = Path(self.annotations_file)
        else:
            for name in [
                "annotations/instances_val2017.json",
                "instances_val2017.json",
            ]:
                path = self.data_path / name
                if path.exists():
                    annotations_path = path
                    break

        if annotations_path and annotations_path.exists():
            self._coco_annotations_file = str(annotations_path)
            self._load_coco_annotations(annotations_path)
        else:
            logger.warning("No COCO annotations file found, using images only")

        self._load_images(images_dir)

        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        logger.info(f"Loaded {len(self._samples)} images")

        if self.use_disk_cache:
            self._preprocessed_dir = self.data_path / "preprocessed_ssd_cache"
            self._preprocessed_dir.mkdir(exist_ok=True)
            self._check_or_create_disk_cache()

        self._is_loaded = True

    def _load_coco_annotations(self, annotations_path: Path) -> None:
        logger.info(f"Loading COCO annotations from {annotations_path}")

        with open(annotations_path, 'r') as f:
            coco = json.load(f)

        # Build image info lookup
        for img in coco.get('images', []):
            self._image_info[img['id']] = {
                'file_name': img['file_name'],
                'width': img.get('width', 0),
                'height': img.get('height', 0),
            }

        # Build annotations lookup by image_id
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self._image_info:
                continue

            # COCO bbox format: [x, y, width, height] in pixels
            bbox = ann['bbox']
            self._annotations[img_id].append({
                'bbox': bbox,
                'category_id': ann['category_id'],
                'iscrowd': ann.get('iscrowd', 0),
                'area': ann.get('area', 0),
            })

        logger.info(f"Loaded annotations for {len(self._annotations)} images, "
                    f"{sum(len(v) for v in self._annotations.values())} objects")

    def _load_images(self, images_dir: Path) -> None:
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

        # If we have annotations, use only images that appear in annotations
        if self._image_info:
            for img_id, info in sorted(self._image_info.items()):
                img_path = images_dir / info['file_name']
                if img_path.exists():
                    self._samples.append({
                        'image_id': img_id,
                        'image_path': str(img_path),
                        'file_name': info['file_name'],
                        'width': info['width'],
                        'height': info['height'],
                    })
                    self._items.append(str(img_id))
        else:
            # Fallback: scan directory
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix in extensions:
                    image_id = int(img_path.stem) if img_path.stem.isdigit() else hash(img_path.stem) % (2**31)
                    self._samples.append({
                        'image_id': image_id,
                        'image_path': str(img_path),
                        'file_name': img_path.name,
                    })
                    self._items.append(str(image_id))

    def _check_or_create_disk_cache(self) -> None:
        if not self._preprocessed_dir:
            return

        cached_files = list(self._preprocessed_dir.glob("*.npy"))
        num_cached = len(cached_files)
        num_samples = len(self._samples)

        if num_cached >= num_samples:
            logger.info(f"Disk cache found: {num_cached} preprocessed files")
            return

        logger.info(f"Creating disk cache: {num_cached}/{num_samples} files exist")
        self._create_disk_cache()

    def _create_disk_cache(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_and_save(idx: int) -> Tuple[bool, Optional[str]]:
            sample = self._samples[idx]
            image_id = sample['image_id']
            cache_path = self._preprocessed_dir / f"{image_id}.npy"

            if cache_path.exists():
                return True, None

            try:
                img_array, _ = self._preprocess_image(sample['image_path'])
                np.save(cache_path, img_array.astype(np.float32))
                return True, None
            except Exception as e:
                return False, f"{image_id}: {e}"

        total = len(self._samples)
        completed = 0
        failed_count = 0

        print(f"Preprocessing {total} COCO images to disk cache...")
        print("(This only happens once, subsequent runs will be fast)")

        num_workers = min(8, total)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_and_save, i): i for i in range(total)}

            for future in as_completed(futures):
                success, error_msg = future.result()
                completed += 1
                if not success and error_msg:
                    logger.warning(f"Failed to preprocess: {error_msg}")
                    failed_count += 1
                if completed % 500 == 0 or completed == total:
                    print(f"\rPreprocessing: {completed}/{total} ({100*completed/total:.1f}%)", end="", flush=True)

        print()
        logger.info(f"Disk cache created: {completed - failed_count}/{total} files")

    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        if self.use_opencv:
            return self._preprocess_image_opencv(image_path)
        else:
            return self._preprocess_image_pil(image_path)

    def _preprocess_image_opencv(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        orig_height, orig_width = img.shape[:2]

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to input size (no aspect ratio preservation per MLPerf reference)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # Normalize: /255 then ImageNet mean/std
        img_array = img.astype(np.float32) / 255.0
        img_array = (img_array - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(NORMALIZE_STD, dtype=np.float32)

        if self.output_layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

        img_array = np.expand_dims(img_array, axis=0)

        preprocess_info = {
            'orig_width': orig_width,
            'orig_height': orig_height,
            'scale_x': self.input_size / orig_width,
            'scale_y': self.input_size / orig_height,
        }

        return img_array, preprocess_info

    def _preprocess_image_pil(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        img = Image.open(image_path).convert('RGB')
        orig_width, orig_height = img.size

        img = img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(NORMALIZE_STD, dtype=np.float32)

        if self.output_layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

        img_array = np.expand_dims(img_array, axis=0)

        preprocess_info = {
            'orig_width': orig_width,
            'orig_height': orig_height,
            'scale_x': self.input_size / orig_width,
            'scale_y': self.input_size / orig_height,
        }

        return img_array, preprocess_info

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def total_count(self) -> int:
        return len(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._is_loaded:
            self.load()

        if self._cache_lock is None:
            import threading
            self._cache_lock = threading.Lock()

        sample = self._samples[index]

        with self._cache_lock:
            if self.cache_preprocessed and index in self._cache:
                return self._cache[index], sample

        # Try disk cache
        if self.use_disk_cache and self._preprocessed_dir:
            cache_path = self._preprocessed_dir / f"{sample['image_id']}.npy"
            if cache_path.exists():
                try:
                    img_array = np.load(cache_path)
                    with self._cache_lock:
                        if self.cache_preprocessed:
                            self._cache[index] = img_array
                    return img_array, sample
                except Exception:
                    cache_path.unlink()

        img_array, preprocess_info = self._preprocess_image(sample['image_path'])

        with self._cache_lock:
            sample['preprocess_info'] = preprocess_info
            if self.cache_preprocessed:
                self._cache[index] = img_array

        if self.use_disk_cache and self._preprocessed_dir:
            cache_path = self._preprocessed_dir / f"{sample['image_id']}.npy"
            try:
                np.save(cache_path, img_array.astype(np.float32))
            except Exception:
                pass

        return img_array, sample

    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        if not self._is_loaded:
            self.load()

        images = []
        sample_infos = []

        for idx in indices:
            img, info = self.get_sample(idx)
            images.append(img[0])
            sample_infos.append(info)

        return np.stack(images), sample_infos

    def get_ground_truth(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        image_id = sample['image_id']
        annotations = self._annotations.get(image_id, [])

        if not annotations:
            return {'boxes': np.array([]), 'labels': np.array([])}

        boxes = []
        labels = []

        for ann in annotations:
            if ann.get('iscrowd', 0):
                continue

            # COCO bbox: [x, y, w, h] in pixels -> normalize to [0, 1]
            bbox = ann['bbox']
            w = sample.get('width', 1)
            h = sample.get('height', 1)
            x_min = bbox[0] / w
            y_min = bbox[1] / h
            x_max = (bbox[0] + bbox[2]) / w
            y_max = (bbox[1] + bbox[3]) / h

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        return {
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.array([], dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64),
        }

    def postprocess(
        self,
        results: Union[np.ndarray, Dict, List],
        indices: List[int]
    ) -> List[Dict[str, np.ndarray]]:
        predictions = []

        if isinstance(results, dict):
            boxes = results.get('boxes', results.get('bboxes', []))
            scores = results.get('scores', [])
            labels = results.get('labels', [])

            for i in range(len(indices)):
                pred = {
                    'boxes': boxes[i] if len(boxes) > i else np.array([]),
                    'scores': scores[i] if len(scores) > i else np.array([]),
                    'labels': labels[i] if len(labels) > i else np.array([]),
                }
                predictions.append(pred)
        elif isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    predictions.append(result)
                else:
                    predictions.append({
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'labels': np.array([]),
                    })
        else:
            for i in range(len(indices)):
                predictions.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([]),
                })

        return predictions

    def compute_accuracy(
        self,
        predictions: List[Dict],
        indices: List[int]
    ) -> Dict[str, float]:
        """Compute mAP using pycocotools if available, else fallback."""
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            if self._coco_annotations_file:
                return self._compute_accuracy_pycocotools(predictions, indices)
        except ImportError:
            logger.warning("pycocotools not available, using simplified mAP")

        return self._compute_accuracy_fallback(predictions, indices)

    def _compute_accuracy_pycocotools(
        self,
        predictions: List[Dict],
        indices: List[int]
    ) -> Dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(self._coco_annotations_file)

        coco_results = []
        for pred, idx in zip(predictions, indices):
            sample = self._samples[idx]
            image_id = sample['image_id']

            boxes = pred.get('boxes', np.array([]))
            scores = pred.get('scores', np.array([]))
            labels = pred.get('labels', np.array([]))

            if len(boxes) == 0:
                continue

            if boxes.ndim == 1 and len(boxes) > 0:
                boxes = boxes.reshape(-1, 4)

            for i in range(len(scores)):
                if i >= len(boxes):
                    break

                box = boxes[i]
                score = float(scores[i])
                label = int(labels[i]) if i < len(labels) else 0

                # Convert to COCO format: [x, y, width, height]
                # Model outputs normalized [0, 1] coordinates (ltrb)
                # Scale directly to original image dimensions per MLCommons reference
                orig_w = sample.get('width', self.input_size)
                orig_h = sample.get('height', self.input_size)

                x1 = float(box[0]) * orig_w
                y1 = float(box[1]) * orig_h
                x2 = float(box[2]) * orig_w
                y2 = float(box[3]) * orig_h

                w = x2 - x1
                h = y2 - y1

                # Map 1-indexed label (1-80) to COCO category ID
                # Matches MLCommons inv_map = [0] + cocoGt.getCatIds()
                label_int = int(label)
                if 1 <= label_int <= 80:
                    category_id = LABEL_TO_COCO_ID[label_int]
                else:
                    continue

                coco_results.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x1, y1, w, h],
                    'score': score,
                })

        if not coco_results:
            logger.warning("No valid predictions for COCO evaluation")
            return {'mAP': 0.0, 'num_samples': len(indices)}

        coco_dt = coco_gt.loadRes(coco_results)

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

        # Evaluate only on images we actually predicted on
        eval_image_ids = [self._samples[idx]['image_id'] for idx in indices]
        coco_eval.params.imgIds = eval_image_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            'mAP': float(coco_eval.stats[0]),        # mAP@0.50:0.95
            'mAP@0.5': float(coco_eval.stats[1]),    # mAP@IoU=0.50
            'mAP@0.75': float(coco_eval.stats[2]),   # mAP@IoU=0.75
            'mAP_small': float(coco_eval.stats[3]),   # mAP for small objects
            'mAP_medium': float(coco_eval.stats[4]),  # mAP for medium objects
            'mAP_large': float(coco_eval.stats[5]),   # mAP for large objects
            'num_predictions': len(coco_results),
            'num_samples': len(indices),
        }

    def _compute_accuracy_fallback(
        self,
        predictions: List[Dict],
        indices: List[int]
    ) -> Dict[str, float]:
        """Simplified mAP computation without pycocotools."""
        ground_truths = [self.get_ground_truth(idx) for idx in indices]

        # Simple AP at IoU=0.5
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred.get('boxes', np.array([]))
            gt_boxes = gt.get('boxes', np.array([]))

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue

            matched = set()
            for pb in pred_boxes:
                best_iou = 0
                best_gt = -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched:
                        continue
                    iou = self._compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi
                if best_iou >= 0.5 and best_gt >= 0:
                    total_tp += 1
                    matched.add(best_gt)
                else:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'mAP': f1,  # Approximate
            'precision': precision,
            'recall': recall,
            'num_samples': len(indices),
        }

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0


class COCOQSL(QuerySampleLibrary):
    """QuerySampleLibrary for COCO 2017 dataset."""

    MAX_CACHE_SIZE = 500  # 1200x1200x3 float32 = 16.5MB each, 500 ~ 8GB

    def __init__(
        self,
        data_path: str,
        annotations_file: Optional[str] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 256,
        input_size: int = INPUT_SIZE,
        output_layout: str = "NHWC",
        use_opencv: bool = True,
        max_cache_size: Optional[int] = None,
    ):
        super().__init__()

        self.dataset = COCODataset(
            data_path=data_path,
            annotations_file=annotations_file,
            count=count,
            input_size=input_size,
            cache_preprocessed=False,  # QSL manages its own LRU cache
            use_disk_cache=True,
            output_layout=output_layout,
            use_opencv=use_opencv,
        )

        self._performance_sample_count = performance_sample_count
        self._max_cache_size = max_cache_size or self.MAX_CACHE_SIZE

        from collections import OrderedDict
        self._loaded_samples: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_lock = None

    def load(self) -> None:
        self.dataset.load()

    @property
    def total_sample_count(self) -> int:
        if not self.dataset._is_loaded:
            self.dataset.load()
        return self.dataset.total_count

    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)

    def _ensure_cache_lock(self):
        if self._cache_lock is None:
            import threading
            self._cache_lock = threading.Lock()

    def _cache_put(self, idx: int, data: np.ndarray) -> None:
        self._ensure_cache_lock()
        with self._cache_lock:
            if idx in self._loaded_samples:
                self._loaded_samples.move_to_end(idx)
                return
            while len(self._loaded_samples) >= self._max_cache_size:
                self._loaded_samples.popitem(last=False)
            self._loaded_samples[idx] = data

    def _cache_get(self, idx: int) -> Optional[np.ndarray]:
        self._ensure_cache_lock()
        with self._cache_lock:
            if idx in self._loaded_samples:
                self._loaded_samples.move_to_end(idx)
                return self._loaded_samples[idx]
            return None

    def load_query_samples(self, sample_indices: List[int]) -> None:
        if not self.dataset._is_loaded:
            self.dataset.load()

        if len(sample_indices) <= self._max_cache_size:
            self._preload_samples(sample_indices)
        else:
            logger.info(f"Lazy loading mode: {len(sample_indices)} samples "
                       f"(cache size: {self._max_cache_size})")

    def _preload_samples(self, sample_indices: List[int]) -> None:
        import os
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed

        to_load = [idx for idx in sample_indices if self._cache_get(idx) is None]

        if not to_load:
            return

        logger.debug(f"Preloading {len(to_load)} samples...")

        def load_single(idx):
            img, _ = self.dataset.get_sample(idx)
            return idx, img

        num_workers = min(os.cpu_count() or 4, len(to_load), 16)
        completed = 0
        total = len(to_load)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single, idx): idx for idx in to_load}

            for future in as_completed(futures):
                try:
                    idx, img = future.result()
                    self._cache_put(idx, img)
                    completed += 1

                    if completed % 100 == 0 or completed == total:
                        pct = completed / total * 100
                        sys.stderr.write(f"\rPreprocessing: {completed}/{total} ({pct:.1f}%)   ")
                        sys.stderr.flush()
                except Exception as e:
                    logger.warning(f"Failed to load sample: {e}")

        if total > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        self._ensure_cache_lock()
        with self._cache_lock:
            for idx in sample_indices:
                self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, np.ndarray]:
        cached = self._cache_get(sample_index)
        if cached is not None:
            return {'input': cached}

        img, _ = self.dataset.get_sample(sample_index)
        self._cache_put(sample_index, img)
        return {'input': img}

    def get_label(self, sample_index: int) -> Dict[str, Any]:
        return self.dataset.get_ground_truth(sample_index)

    def get_sample_to_image_id_mapping(self) -> Dict[int, int]:
        """Get mapping from sample index to COCO image ID."""
        if not self.dataset._is_loaded:
            self.dataset.load()
        return {idx: sample['image_id'] for idx, sample in enumerate(self.dataset._samples)}

    def get_sample_to_filename_mapping(self) -> Dict[int, str]:
        """Get mapping from sample index to image filename."""
        if not self.dataset._is_loaded:
            self.dataset.load()
        return {idx: sample.get('file_name', '') for idx, sample in enumerate(self.dataset._samples)}
