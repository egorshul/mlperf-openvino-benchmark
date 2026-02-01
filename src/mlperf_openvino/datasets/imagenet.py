"""ImageNet dataset for ResNet50 benchmark."""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from ..core.config import PreprocessingConfig

logger = logging.getLogger(__name__)


class ImageNetDataset(BaseDataset):
    """
    ImageNet validation dataset for image classification.

    Expected directory structure:
        data_path/
            val/
                ILSVRC2012_val_00000001.JPEG
                ...
            val_map.txt  # Format: filename label
    """

    def __init__(
        self,
        data_path: str,
        val_map_path: Optional[str] = None,
        preprocessing: Optional[PreprocessingConfig] = None,
        count: Optional[int] = None,
        cache_preprocessed: bool = True,
        use_opencv: bool = True,
        **kwargs
    ):
        """Initialize ImageNet dataset."""
        if not PIL_AVAILABLE and not CV2_AVAILABLE:
            raise ImportError("Either Pillow or OpenCV is required")

        super().__init__(data_path, count, **kwargs)

        self.val_map_path = val_map_path
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.cache_preprocessed = cache_preprocessed
        self.use_opencv = use_opencv and CV2_AVAILABLE

        self._image_paths: List[str] = []
        self._preprocessed_cache: Dict[int, np.ndarray] = {}

    def load(self) -> None:
        """Load the dataset metadata."""
        if self._loaded:
            logger.warning("Dataset already loaded")
            return

        data_path = Path(self.data_path)

        if self.val_map_path:
            val_map_path = Path(self.val_map_path)
        else:
            val_map_path = data_path / "val_map.txt"

        if val_map_path.exists():
            logger.info(f"Loading validation map from {val_map_path}")
            self._load_from_val_map(val_map_path, data_path)
        else:
            logger.info("No val_map.txt found, scanning directory...")
            self._load_from_directory(data_path)

        logger.info(f"Loaded {len(self._items)} images")
        self._loaded = True

    def _load_from_val_map(self, val_map_path: Path, data_path: Path) -> None:
        """Load dataset from val_map.txt file."""
        with open(val_map_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])

                    img_path = data_path / filename
                    if not img_path.exists():
                        img_path = data_path / "val" / filename

                    if img_path.exists():
                        self._items.append(str(img_path))
                        self._image_paths.append(str(img_path))
                        self._labels.append(label)

    def _load_from_directory(self, data_path: Path) -> None:
        """Load dataset by scanning directory."""
        val_dir = data_path / "val" if (data_path / "val").exists() else data_path

        extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

        for img_file in sorted(val_dir.iterdir()):
            if img_file.suffix in extensions:
                self._items.append(str(img_file))
                self._image_paths.append(str(img_file))
                # Without val_map, we don't have labels
                self._labels.append(-1)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image following MLCommons reference (pre_process_vgg)."""
        if self.use_opencv:
            return self._preprocess_image_opencv(image_path)
        else:
            return self._preprocess_image_pil(image_path)

    def _preprocess_image_opencv(self, image_path: str) -> np.ndarray:
        """OpenCV implementation - faster preprocessing."""
        # OpenCV loads as BGR
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # MLCommons reference uses RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.preprocessing and self.preprocessing.center_crop:
            crop_h, crop_w = self.preprocessing.center_crop
        else:
            crop_h, crop_w = 224, 224

        # MLCommons uses scale=87.5%, so target_size / 0.875 ~ 256 for 224
        h, w = img.shape[:2]
        scale = 87.5
        new_height = int(100.0 * crop_h / scale)  # 256 for crop_h=224
        new_width = int(100.0 * crop_w / scale)   # 256 for crop_w=224

        if h > w:
            new_w = new_width
            new_h = int(new_height * h / w)
        else:
            new_h = new_height
            new_w = int(new_width * w / h)

        # cv2.INTER_AREA is best for downscaling (matches MLCommons reference)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img[top:top + crop_h, left:left + crop_w]

        img_array = img.astype(np.float32)

        if self.preprocessing.channel_order == "BGR":
            img_array = img_array[:, :, ::-1].copy()

        mean = np.array(self.preprocessing.mean, dtype=np.float32)
        img_array = img_array - mean

        std = np.array(self.preprocessing.std, dtype=np.float32)
        img_array = img_array / std

        output_layout = getattr(self.preprocessing, 'output_layout', 'NHWC')
        if output_layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _preprocess_image_pil(self, image_path: str) -> np.ndarray:
        """PIL implementation - fallback when OpenCV unavailable."""
        img = Image.open(image_path).convert("RGB")

        if self.preprocessing.center_crop:
            crop_h, crop_w = self.preprocessing.center_crop
        else:
            crop_h, crop_w = 224, 224

        # MLCommons uses scale=87.5%, so target_size / 0.875 ~ 256 for 224
        w, h = img.size
        scale = 87.5
        new_height = int(100.0 * crop_h / scale)  # 256 for crop_h=224
        new_width = int(100.0 * crop_w / scale)   # 256 for crop_w=224

        if h > w:
            new_w = new_width
            new_h = int(new_height * h / w)
        else:
            new_h = new_height
            new_w = int(new_width * w / h)

        # BOX for downscaling - closest to cv2.INTER_AREA used by MLCommons
        img = img.resize((new_w, new_h), Image.Resampling.BOX)

        w, h = img.size
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))

        img_array = np.array(img, dtype=np.float32)

        if self.preprocessing.channel_order == "BGR":
            img_array = img_array[:, :, ::-1]

        mean = np.array(self.preprocessing.mean, dtype=np.float32)
        img_array = img_array - mean

        std = np.array(self.preprocessing.std, dtype=np.float32)
        img_array = img_array / std

        # NHWC is default, model handles conversion via PrePostProcessor
        output_layout = getattr(self.preprocessing, 'output_layout', 'NHWC')
        if output_layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
            img_array = np.expand_dims(img_array, axis=0)    # -> NCHW
        else:
            img_array = np.expand_dims(img_array, axis=0)    # -> NHWC

        return img_array

    def get_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Get a preprocessed sample by index."""
        if not self._loaded:
            self.load()

        if self.cache_preprocessed and index in self._preprocessed_cache:
            return self._preprocessed_cache[index], self._labels[index]

        img_path = self._image_paths[index]
        preprocessed = self._preprocess_image(img_path)

        if self.cache_preprocessed:
            self._preprocessed_cache[index] = preprocessed

        return preprocessed, self._labels[index]

    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Get multiple preprocessed samples."""
        if not self._loaded:
            self.load()

        samples = []
        labels = []

        for idx in indices:
            sample, label = self.get_sample(idx)
            samples.append(sample)
            labels.append(label)

        batch = np.concatenate(samples, axis=0)

        return batch, labels

    def postprocess(
        self,
        results: np.ndarray,
        indices: List[int]
    ) -> List[int]:
        """Postprocess inference results to get predicted classes."""
        predictions = []

        if results.dtype in (np.int64, np.int32):
            # Model already computed argmax (e.g., ArgMax:0 output)
            # MLPerf ResNet50 ONNX uses 1-based indexing (1-1000 for ImageNet)
            results = results.flatten()
            for idx in results:
                pred = int(idx) - 1  # MLCommons offset=-1
                pred = max(0, min(999, pred))
                predictions.append(pred)
        else:
            if len(results.shape) == 1:
                results = results.reshape(1, -1)

            num_classes = results.shape[1]

            # Single value per sample = already argmax'd class index (stored as float)
            if num_classes == 1:
                for i in range(results.shape[0]):
                    pred = int(results[i, 0]) - 1  # MLCommons offset=-1
                    pred = max(0, min(999, pred))
                    predictions.append(pred)
            else:
                # Full logits/probabilities (1001 classes), compute argmax
                for i in range(results.shape[0]):
                    pred = int(np.argmax(results[i])) - 1  # MLCommons offset=-1
                    pred = max(0, min(999, pred))
                    predictions.append(pred)

        return predictions

    def compute_accuracy(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """Compute Top-1 accuracy."""
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length")

        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        total = len(predictions)

        return {
            "top1_accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }


class ImageNetQSL(QuerySampleLibrary):
    """ImageNet Query Sample Library for MLPerf LoadGen."""

    def __init__(
        self,
        data_path: str,
        val_map_path: Optional[str] = None,
        preprocessing: Optional[PreprocessingConfig] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 1024,
        **kwargs
    ):
        """Initialize ImageNet QSL."""
        self._dataset = ImageNetDataset(
            data_path=data_path,
            val_map_path=val_map_path,
            preprocessing=preprocessing,
            count=count,
            cache_preprocessed=True,
            **kwargs
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, np.ndarray] = {}

    def load(self) -> None:
        """Load the dataset."""
        self._dataset.load()

    def load_query_samples(self, sample_list: List[int]) -> None:
        """Load samples into memory with parallel preprocessing."""
        if not self._dataset.is_loaded:
            self._dataset.load()

        to_load = [idx for idx in sample_list if idx not in self._loaded_samples]

        if not to_load:
            return

        logger.debug(f"Loading {len(to_load)} ImageNet samples with parallel preprocessing...")

        import os
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def load_single(idx):
            data, _ = self._dataset.get_sample(idx)
            return idx, data

        num_workers = min(os.cpu_count() or 4, len(to_load), 16)
        completed = 0
        total = len(to_load)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single, idx): idx for idx in to_load}

            for future in as_completed(futures):
                try:
                    idx, data = future.result()
                    self._loaded_samples[idx] = data
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

    def unload_query_samples(self, sample_list: List[int]) -> None:
        """Unload samples from memory."""
        for sample_id in sample_list:
            if sample_id in self._loaded_samples:
                del self._loaded_samples[sample_id]

    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        """Get features for a sample."""
        if sample_id in self._loaded_samples:
            data = self._loaded_samples[sample_id]
        else:
            data, _ = self._dataset.get_sample(sample_id)

        return {"input": data}

    def get_label(self, sample_id: int) -> int:
        """Get label for a sample."""
        return self._dataset._labels[sample_id]

    @property
    def total_sample_count(self) -> int:
        """Get total number of samples."""
        if not self._dataset.is_loaded:
            self._dataset.load()
        return self._dataset.sample_count

    @property
    def performance_sample_count(self) -> int:
        """Get number of samples for performance testing."""
        return min(self._performance_sample_count, self.total_sample_count)

    @property
    def dataset(self) -> ImageNetDataset:
        """Get underlying dataset."""
        return self._dataset
