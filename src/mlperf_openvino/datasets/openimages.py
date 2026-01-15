"""
OpenImages dataset for RetinaNet Object Detection benchmark.

This module provides dataset handling for the OpenImages dataset
used in MLPerf Inference for RetinaNet model evaluation.
"""

import csv
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

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# RetinaNet preprocessing constants
INPUT_SIZE = 800
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# COCO classes used in MLPerf (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation.

    Args:
        recalls: Array of recall values
        precisions: Array of precision values

    Returns:
        Average Precision value
    """
    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        if len(p) > 0:
            ap += np.max(p)

    return ap / 101


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 601  # OpenImages has up to 601 classes
) -> Dict[str, float]:
    """
    Compute mean Average Precision.

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes (601 for OpenImages)

    Returns:
        Dictionary with mAP and per-class AP
    """
    # Organize detections and ground truths by class
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)
    all_classes = set()  # Track all classes we see

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Detections
        if pred is not None and len(pred.get('boxes', [])) > 0:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                class_id = int(label)
                all_classes.add(class_id)
                all_detections[class_id].append({
                    'image_idx': img_idx,
                    'box': box,
                    'score': score,
                })

        # Ground truths
        if gt is not None and len(gt.get('boxes', [])) > 0:
            for box, label in zip(gt['boxes'], gt['labels']):
                class_id = int(label)
                all_classes.add(class_id)
                all_ground_truths[class_id].append({
                    'image_idx': img_idx,
                    'box': box,
                    'matched': False,
                })

    # Compute AP for each class that has ground truths
    aps = {}
    for class_id in all_classes:
        detections = all_detections[class_id]
        gts = all_ground_truths[class_id]

        if len(gts) == 0:
            continue

        # Sort detections by score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        # Reset matched flags
        for gt in gts:
            gt['matched'] = False

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for det_idx, det in enumerate(detections):
            img_gts = [g for g in gts if g['image_idx'] == det['image_idx'] and not g['matched']]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(img_gts):
                iou = compute_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[det_idx] = 1
                img_gts[best_gt_idx]['matched'] = True
            else:
                fp[det_idx] = 1

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        aps[class_id] = compute_ap(recalls, precisions)

    # Compute mAP
    if aps:
        map_value = np.mean(list(aps.values()))
    else:
        map_value = 0.0

    return {
        'mAP': map_value,
        'mAP@0.5': map_value,
        'per_class_ap': aps,
        'num_classes_evaluated': len(aps),
    }


class OpenImagesDataset(BaseDataset):
    """
    OpenImages dataset for RetinaNet Object Detection benchmark.

    Expected directory structure:
        data_path/
        ├── images/
        │   ├── image1.jpg
        │   └── ...
        └── annotations/
            └── validation-annotations-bbox.csv
    """

    def __init__(
        self,
        data_path: str,
        annotations_file: Optional[str] = None,
        count: Optional[int] = None,
        input_size: int = INPUT_SIZE,
        cache_preprocessed: bool = True,
        use_disk_cache: bool = True,  # Use preprocessed numpy files
    ):
        """
        Initialize OpenImages dataset.

        Args:
            data_path: Path to dataset directory
            annotations_file: Path to annotations CSV file
            count: Number of samples to use (None = all)
            input_size: Input image size
            cache_preprocessed: Whether to cache preprocessed images in memory
            use_disk_cache: Whether to use/create preprocessed numpy cache on disk
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.annotations_file = annotations_file
        self.input_size = input_size
        self.cache_preprocessed = cache_preprocessed
        self.use_disk_cache = use_disk_cache

        self._samples: List[Dict[str, Any]] = []
        self._annotations: Dict[str, List[Dict]] = defaultdict(list)
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_lock = None  # Lazy init for thread safety
        self._is_loaded = False
        self._preprocessed_dir: Optional[Path] = None

    def load(self) -> None:
        """Load dataset metadata and annotations."""
        if self._is_loaded:
            return

        logger.info(f"Loading OpenImages dataset from {self.data_path}")

        # Find images directory
        images_dir = self.data_path / "images"
        if not images_dir.exists():
            images_dir = self.data_path / "validation" / "data"
        if not images_dir.exists():
            images_dir = self.data_path

        # Find annotations file (prefer COCO JSON over CSV)
        annotations_path = None
        if self.annotations_file:
            annotations_path = Path(self.annotations_file)
        else:
            # Prefer COCO JSON format (has correct class mapping)
            for name in [
                "annotations/openimages-mlperf.json",
                "openimages-mlperf.json",
                "annotations/validation-annotations-bbox.csv",
                "validation-annotations-bbox.csv",
            ]:
                path = self.data_path / name
                if path.exists():
                    annotations_path = path
                    break

        # Load annotations
        if annotations_path and annotations_path.exists():
            if annotations_path.suffix == '.json':
                self._load_coco_annotations(annotations_path)
            else:
                self._load_annotations(annotations_path)
        else:
            logger.warning("No annotations file found, using images only")

        # Load image list
        self._load_images(images_dir)

        # Limit count if specified
        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        logger.info(f"Loaded {len(self._samples)} images")

        # Set up preprocessed cache directory
        if self.use_disk_cache:
            self._preprocessed_dir = self.data_path / "preprocessed_cache"
            self._preprocessed_dir.mkdir(exist_ok=True)
            self._check_or_create_disk_cache()

        self._is_loaded = True

    def _check_or_create_disk_cache(self) -> None:
        """Check if disk cache exists, create if needed."""
        if not self._preprocessed_dir:
            return

        # Check how many cached files exist
        cached_files = list(self._preprocessed_dir.glob("*.npy"))
        num_cached = len(cached_files)
        num_samples = len(self._samples)

        if num_cached >= num_samples:
            logger.info(f"Disk cache found: {num_cached} preprocessed files")
            return

        logger.info(f"Creating disk cache: {num_cached}/{num_samples} files exist")
        self._create_disk_cache()

    def _create_disk_cache(self) -> None:
        """Preprocess all images and save to disk (like NVIDIA submissions)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        failed_images = []

        def process_and_save(idx: int) -> Tuple[bool, Optional[str]]:
            sample = self._samples[idx]
            image_id = sample['image_id']
            image_path = sample['image_path']
            cache_path = self._preprocessed_dir / f"{image_id}.npy"

            if cache_path.exists():
                return True, None

            try:
                img_array, _ = self._preprocess_image(image_path)
                np.save(cache_path, img_array.astype(np.float32))
                return True, None
            except Exception as e:
                # Check if image file is corrupted/empty
                try:
                    file_size = Path(image_path).stat().st_size
                    if file_size < 1000:  # Likely corrupted if < 1KB
                        return False, f"{image_id} (corrupted, {file_size} bytes)"
                except:
                    pass
                return False, f"{image_id}: {e}"

        total = len(self._samples)
        completed = 0
        failed_count = 0

        print(f"Preprocessing {total} images to disk cache...")
        print("(This only happens once, subsequent runs will be fast)")

        num_workers = min(8, total)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_and_save, i): i for i in range(total)}

            for future in as_completed(futures):
                success, error_msg = future.result()
                completed += 1
                if not success and error_msg:
                    failed_images.append(error_msg)
                    failed_count += 1
                if completed % 500 == 0 or completed == total:
                    print(f"\rPreprocessing: {completed}/{total} ({100*completed/total:.1f}%)", end="", flush=True)

        print()

        if failed_images:
            logger.warning(f"Failed to preprocess {failed_count} images (corrupted downloads)")
            # Auto-fix: re-download and preprocess corrupted images
            fixed = self._auto_fix_corrupted_images(failed_images)
            failed_count -= fixed

        logger.info(f"Disk cache created: {completed - failed_count}/{total} files")

    def _auto_fix_corrupted_images(self, failed_list: List[str]) -> int:
        """Automatically re-download and preprocess corrupted images."""
        # Extract image IDs from error messages
        image_ids = []
        for msg in failed_list:
            # Format: "image_id (corrupted...)" or "image_id: error"
            image_id = msg.split()[0].split(':')[0].split('(')[0].strip()
            if image_id:
                image_ids.append(image_id)

        if not image_ids:
            return 0

        print(f"\nAuto-fixing {len(image_ids)} corrupted images...")

        # Find images directory
        images_dir = self.data_path / "validation" / "data"
        if not images_dir.exists():
            images_dir = self.data_path / "images"
        if not images_dir.exists():
            logger.warning("Cannot find images directory for auto-fix")
            return 0

        # Delete corrupted files
        for image_id in image_ids:
            img_path = images_dir / f"{image_id}.jpg"
            if img_path.exists():
                try:
                    img_path.unlink()
                except:
                    pass

        # Re-download from S3
        try:
            from ..utils.dataset_downloader import _download_openimages_from_s3
            print(f"Re-downloading {len(image_ids)} images from S3...")
            _download_openimages_from_s3(image_ids, images_dir, num_workers=4)

            # Re-preprocess the fixed images
            print("Re-preprocessing fixed images...")
            fixed_count = 0
            for image_id in image_ids:
                img_path = images_dir / f"{image_id}.jpg"
                cache_path = self._preprocessed_dir / f"{image_id}.npy"

                if img_path.exists():
                    try:
                        img_array, _ = self._preprocess_image(img_path)
                        np.save(cache_path, img_array.astype(np.float32))
                        fixed_count += 1
                    except Exception as e:
                        logger.warning(f"Still failed: {image_id}")

            print(f"Auto-fixed {fixed_count}/{len(image_ids)} images")
            return fixed_count

        except Exception as e:
            logger.warning(f"Auto-fix failed: {e}")
            return 0

    def _load_coco_annotations(self, annotations_path: Path) -> None:
        """Load annotations from COCO JSON file (preferred format with class mapping)."""
        import json

        logger.info(f"Loading COCO annotations from {annotations_path}")

        with open(annotations_path, 'r') as f:
            coco = json.load(f)

        # Build category mapping
        self._categories = {}
        for cat in coco.get('categories', []):
            self._categories[cat['id']] = cat['name']

        logger.info(f"Loaded {len(self._categories)} categories")

        # Build image id to filename mapping
        image_info = {}
        for img in coco.get('images', []):
            image_info[img['id']] = {
                'file_name': img['file_name'],
                'width': img.get('width', 800),
                'height': img.get('height', 600),
            }

        # Load annotations grouped by image
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_info:
                continue

            info = image_info[img_id]
            image_id = info['file_name'].replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            width, height = info['width'], info['height']

            # COCO bbox is [x, y, width, height] in pixels
            bbox = ann['bbox']
            x_min = bbox[0] / width
            y_min = bbox[1] / height
            x_max = (bbox[0] + bbox[2]) / width
            y_max = (bbox[1] + bbox[3]) / height

            self._annotations[image_id].append({
                'box': [x_min, y_min, x_max, y_max],  # Normalized
                'category_id': ann['category_id'],  # Proper class ID!
                'label_name': self._categories.get(ann['category_id'], ''),
                'is_group_of': ann.get('iscrowd', 0) == 1,
                'is_occluded': False,
                'is_truncated': False,
            })

        logger.info(f"Loaded annotations for {len(self._annotations)} images")

    def _load_annotations(self, annotations_path: Path) -> None:
        """Load annotations from CSV file."""
        logger.info(f"Loading annotations from {annotations_path}")

        with open(annotations_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_id = row.get('ImageID', row.get('image_id', ''))

                # Parse bounding box (normalized coordinates)
                try:
                    x_min = float(row.get('XMin', row.get('x_min', 0)))
                    x_max = float(row.get('XMax', row.get('x_max', 0)))
                    y_min = float(row.get('YMin', row.get('y_min', 0)))
                    y_max = float(row.get('YMax', row.get('y_max', 0)))
                except (ValueError, TypeError):
                    continue

                label_name = row.get('LabelName', row.get('label', ''))

                self._annotations[image_id].append({
                    'box': [x_min, y_min, x_max, y_max],  # Normalized
                    'label_name': label_name,
                    'is_group_of': row.get('IsGroupOf', '0') == '1',
                    'is_occluded': row.get('IsOccluded', '0') == '1',
                    'is_truncated': row.get('IsTruncated', '0') == '1',
                })

        logger.info(f"Loaded annotations for {len(self._annotations)} images")

    def _load_images(self, images_dir: Path) -> None:
        """Load image list from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix in extensions:
                image_id = img_path.stem

                self._samples.append({
                    'image_id': image_id,
                    'image_path': str(img_path),
                    'annotations': self._annotations.get(image_id, []),
                })
                self._items.append(image_id)

    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for RetinaNet.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
        """
        img = Image.open(image_path).convert('RGB')
        orig_width, orig_height = img.size

        # Calculate scale to resize shortest side to input_size
        scale = self.input_size / min(orig_width, orig_height)

        # Resize
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Pad to square
        padded = Image.new('RGB', (self.input_size, self.input_size), (128, 128, 128))
        paste_x = (self.input_size - new_width) // 2
        paste_y = (self.input_size - new_height) // 2
        padded.paste(img, (paste_x, paste_y))

        # Convert to numpy and normalize
        img_array = np.array(padded, dtype=np.float32) / 255.0

        # Normalize with ImageNet mean/std
        mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
        std = np.array(NORMALIZE_STD, dtype=np.float32)
        img_array = (img_array - mean) / std

        # Convert to NCHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        preprocess_info = {
            'orig_width': orig_width,
            'orig_height': orig_height,
            'scale': scale,
            'pad_x': paste_x,
            'pad_y': paste_y,
        }

        return img_array, preprocess_info

    def _fix_and_preprocess(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Re-download corrupted image and preprocess it."""
        image_id = sample['image_id']
        image_path = Path(sample['image_path'])

        # Find images directory
        images_dir = image_path.parent

        # Delete corrupted file
        if image_path.exists():
            try:
                old_size = image_path.stat().st_size
                logger.info(f"Deleting corrupted {image_id} ({old_size} bytes)")
                image_path.unlink()
            except:
                pass

        # Re-download from S3 with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from ..utils.dataset_downloader import _download_openimages_from_s3
                _download_openimages_from_s3([image_id], images_dir, num_workers=1)

                # Verify download
                if image_path.exists():
                    file_size = image_path.stat().st_size
                    if file_size > 1000:  # Valid image should be > 1KB
                        # Try to open with PIL to verify
                        try:
                            with Image.open(image_path) as img:
                                img.verify()
                            logger.info(f"Successfully re-downloaded {image_id} ({file_size} bytes)")
                            break
                        except Exception:
                            logger.warning(f"Downloaded file still corrupted, retry {attempt+1}/{max_retries}")
                            image_path.unlink()
                    else:
                        logger.warning(f"Downloaded file too small ({file_size} bytes), retry {attempt+1}/{max_retries}")
                        image_path.unlink()
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed: {e}")

            if attempt == max_retries - 1:
                logger.error(f"Failed to download valid {image_id} after {max_retries} attempts")
                dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
                return dummy, {}

        # Try preprocessing again
        try:
            return self._preprocess_image(image_path)
        except Exception as e:
            logger.error(f"Still failed after re-download {image_id}: {e}")
            # Return dummy image
            dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            return dummy, {}

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def total_count(self) -> int:
        return len(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get preprocessed sample (thread-safe).

        Uses disk cache (numpy files) if available for fast loading.
        Auto-fixes truncated/corrupted images by re-downloading.

        Args:
            index: Sample index

        Returns:
            Tuple of (preprocessed_image, sample_info)
        """
        if not self._is_loaded:
            self.load()

        # Lazy init lock for thread safety during parallel loading
        if self._cache_lock is None:
            import threading
            self._cache_lock = threading.Lock()

        sample = self._samples[index]

        # Check memory cache first
        with self._cache_lock:
            if self.cache_preprocessed and index in self._cache:
                return self._cache[index], sample

        # Try disk cache (fast numpy load)
        if self.use_disk_cache and self._preprocessed_dir:
            cache_path = self._preprocessed_dir / f"{sample['image_id']}.npy"
            if cache_path.exists():
                try:
                    img_array = np.load(cache_path)
                    # Store in memory cache
                    with self._cache_lock:
                        if self.cache_preprocessed:
                            self._cache[index] = img_array
                    return img_array, sample
                except Exception:
                    # Corrupted cache file, remove it
                    cache_path.unlink()

        # Fallback: preprocess from original image (with auto-fix for corrupted)
        try:
            img_array, preprocess_info = self._preprocess_image(sample['image_path'])
        except (OSError, IOError) as e:
            # Image is truncated/corrupted - try to re-download
            logger.warning(f"Corrupted image {sample['image_id']}, attempting auto-fix...")
            img_array, preprocess_info = self._fix_and_preprocess(sample)

        # Store in caches
        with self._cache_lock:
            sample['preprocess_info'] = preprocess_info
            if self.cache_preprocessed:
                self._cache[index] = img_array

        # Also save to disk cache for next time
        if self.use_disk_cache and self._preprocessed_dir:
            cache_path = self._preprocessed_dir / f"{sample['image_id']}.npy"
            try:
                np.save(cache_path, img_array.astype(np.float32))
            except Exception:
                pass  # Non-critical, ignore errors

        return img_array, sample

    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        """Get batch of samples."""
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
        """
        Get ground truth annotations for a sample.

        Args:
            index: Sample index

        Returns:
            Dictionary with 'boxes' (normalized) and 'labels'
        """
        sample = self._samples[index]
        annotations = sample.get('annotations', [])

        if not annotations:
            return {'boxes': np.array([]), 'labels': np.array([])}

        boxes = []
        labels = []

        for ann in annotations:
            if ann.get('is_group_of', False):
                continue

            # Box is in normalized coords [x_min, y_min, x_max, y_max]
            box = ann['box']
            boxes.append(box)

            # Get category_id (from COCO JSON) or default to 0
            category_id = ann.get('category_id', 0)
            labels.append(category_id)

        return {
            'boxes': np.array(boxes) if boxes else np.array([]),
            'labels': np.array(labels) if labels else np.array([]),
        }

    def postprocess(
        self,
        results: Union[np.ndarray, Dict, List],
        indices: List[int]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Postprocess RetinaNet outputs.

        Args:
            results: Model outputs
            indices: Sample indices

        Returns:
            List of detection dicts with 'boxes', 'scores', 'labels'
        """
        predictions = []

        # Handle different output formats
        if isinstance(results, dict):
            # Single dict with batched outputs
            boxes = results.get('boxes', results.get('detection_boxes', []))
            scores = results.get('scores', results.get('detection_scores', []))
            labels = results.get('labels', results.get('detection_classes', []))

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
            # Assume numpy array
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
        """
        Compute mAP accuracy.

        Args:
            predictions: List of prediction dicts
            indices: Sample indices

        Returns:
            Dictionary with mAP and other metrics
        """
        ground_truths = [self.get_ground_truth(idx) for idx in indices]

        return compute_map(predictions, ground_truths)


class OpenImagesQSL(QuerySampleLibrary):
    """
    Query Sample Library for OpenImages dataset.

    Implements the MLPerf LoadGen QSL interface for RetinaNet benchmark.
    """

    def __init__(
        self,
        data_path: str,
        annotations_file: Optional[str] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 24576,  # MLPerf default
        input_size: int = INPUT_SIZE,
    ):
        """
        Initialize OpenImages QSL.

        Args:
            data_path: Path to dataset directory
            annotations_file: Path to annotations file
            count: Number of samples to use
            performance_sample_count: Number of samples for performance run
            input_size: Input image size
        """
        super().__init__()

        self.dataset = OpenImagesDataset(
            data_path=data_path,
            annotations_file=annotations_file,
            count=count,
            input_size=input_size,
            cache_preprocessed=True,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, np.ndarray] = {}

    def load(self) -> None:
        """Load the dataset."""
        self.dataset.load()

    @property
    def total_sample_count(self) -> int:
        if not self.dataset._is_loaded:
            self.dataset.load()
        return self.dataset.total_count

    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)

    def load_query_samples(self, sample_indices: List[int]) -> None:
        """Load samples into memory with progress tracking and parallel loading."""
        if not self.dataset._is_loaded:
            self.dataset.load()

        # Filter indices that need to be loaded
        to_load = [idx for idx in sample_indices if idx not in self._loaded_samples]

        if not to_load:
            return

        logger.info(f"Loading {len(to_load)} OpenImages samples into memory...")

        # Use parallel loading for faster preprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import sys

        def load_single(idx):
            img, _ = self.dataset.get_sample(idx)
            return idx, img

        num_workers = min(8, len(to_load))
        completed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single, idx): idx for idx in to_load}

            for future in as_completed(futures):
                idx, img = future.result()
                self._loaded_samples[idx] = img
                completed += 1

                # Progress update every 500 samples
                if completed % 500 == 0 or completed == len(to_load):
                    print(f"\rLoading samples: {completed}/{len(to_load)} ({100*completed/len(to_load):.1f}%)", end="", flush=True)

        print()  # Newline after progress
        logger.info(f"Loaded {len(to_load)} samples into memory")

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """Unload samples from memory."""
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, np.ndarray]:
        """Get input features for a sample."""
        if sample_index in self._loaded_samples:
            return {'input': self._loaded_samples[sample_index]}

        img, _ = self.dataset.get_sample(sample_index)
        return {'input': img}

    def get_label(self, sample_index: int) -> Dict[str, Any]:
        """Get ground truth for a sample."""
        return self.dataset.get_ground_truth(sample_index)
