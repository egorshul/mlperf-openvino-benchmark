"""
ImageNet dataset for ResNet50 benchmark.
"""

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
        **kwargs
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            data_path: Path to ImageNet validation images
            val_map_path: Path to validation map file
            preprocessing: Preprocessing configuration
            count: Number of samples to use
            cache_preprocessed: Whether to cache preprocessed images
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for image loading. Install with: pip install Pillow")
        
        super().__init__(data_path, count, **kwargs)
        
        self.val_map_path = val_map_path
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.cache_preprocessed = cache_preprocessed
        
        self._image_paths: List[str] = []
        self._preprocessed_cache: Dict[int, np.ndarray] = {}
    
    def load(self) -> None:
        """Load the dataset metadata."""
        if self._loaded:
            logger.warning("Dataset already loaded")
            return
        
        data_path = Path(self.data_path)
        
        # Find val_map.txt
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
                    
                    # Try to find the image
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
        """
        Preprocess a single image following MLCommons reference (pre_process_vgg).

        Args:
            image_path: Path to the image

        Returns:
            Preprocessed image as numpy array (NCHW format)
        """
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Get target dimensions from center_crop (default 224x224)
        if self.preprocessing.center_crop:
            crop_h, crop_w = self.preprocessing.center_crop
        else:
            crop_h, crop_w = 224, 224

        # Resize with aspect ratio preservation (MLCommons reference)
        # Scale so that after center crop we get the target size
        # MLCommons uses scale=87.5%, so target_size / 0.875 ≈ 256 for 224
        w, h = img.size
        scale = 87.5
        new_height = int(100.0 * crop_h / scale)  # 256 for crop_h=224
        new_width = int(100.0 * crop_w / scale)   # 256 for crop_w=224

        if h > w:
            # Width is smaller, scale based on width
            new_w = new_width
            new_h = int(new_height * h / w)
        else:
            # Height is smaller, scale based on height
            new_h = new_height
            new_w = int(new_width * w / h)

        # Resize preserving aspect ratio (use LANCZOS for quality, similar to INTER_AREA)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        w, h = img.size
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))

        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)

        # Channel order (already RGB from PIL)
        if self.preprocessing.channel_order == "BGR":
            img_array = img_array[:, :, ::-1]

        # Apply mean subtraction
        mean = np.array(self.preprocessing.mean, dtype=np.float32)
        img_array = img_array - mean

        # Apply std normalization
        std = np.array(self.preprocessing.std, dtype=np.float32)
        img_array = img_array / std

        # Convert to target layout
        output_layout = getattr(self.preprocessing, 'output_layout', 'NCHW')
        if output_layout == "NCHW":
            # HWC -> CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            # Add batch dimension -> NCHW
            img_array = np.expand_dims(img_array, axis=0)
        else:
            # Keep HWC, add batch dimension -> NHWC
            img_array = np.expand_dims(img_array, axis=0)

        return img_array
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Get a preprocessed sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
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
        """
        Get multiple preprocessed samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (batch_data, labels)
        """
        if not self._loaded:
            self.load()
        
        samples = []
        labels = []
        
        for idx in indices:
            sample, label = self.get_sample(idx)
            samples.append(sample)
            labels.append(label)
        
        # Concatenate along batch dimension
        batch = np.concatenate(samples, axis=0)
        
        return batch, labels
    
    def postprocess(
        self,
        results: np.ndarray,
        indices: List[int]
    ) -> List[int]:
        """
        Postprocess inference results to get predicted classes.

        Args:
            results: Raw inference output - can be:
                - Pre-computed argmax indices (shape [N] or [N,1], dtype int64/int32/float with single value)
                - Logits/probabilities (shape [N, num_classes], dtype float32)
            indices: Sample indices (unused here)

        Returns:
            List of predicted class indices (0-999 for ImageNet)
        """
        predictions = []

        # Handle different output formats
        if results.dtype in (np.int64, np.int32):
            # Model already computed argmax (e.g., ArgMax:0 output)
            # Results contain class indices directly
            results = results.flatten()
            for idx in results:
                # MLPerf ResNet50 ONNX model uses 1001 classes (0=background, 1-1000=ImageNet)
                # Subtract 1 to convert to 0-999 range for val_map.txt labels
                pred = int(idx) - 1
                # Clamp to valid range (in case of background prediction)
                pred = max(0, min(999, pred))
                predictions.append(pred)
        else:
            # Float output - check if it's a single value (class index stored as float)
            # or full logits/probabilities array
            if len(results.shape) == 1:
                results = results.reshape(1, -1)

            num_classes = results.shape[1]

            # Single value per sample = already argmax'd class index (stored as float)
            if num_classes == 1:
                for i in range(results.shape[0]):
                    # This is the class index stored as float
                    pred = int(results[i, 0])
                    # MLPerf ResNet50 ONNX model uses 1001 classes (0=background, 1-1000=ImageNet)
                    pred = pred - 1
                    pred = max(0, min(999, pred))
                    predictions.append(pred)
            else:
                # Full logits/probabilities, need to compute argmax
                for i in range(results.shape[0]):
                    pred = int(np.argmax(results[i]))

                    # If model has 1001 classes (with background), subtract 1
                    if num_classes == 1001:
                        pred = pred - 1
                        pred = max(0, min(999, pred))

                    predictions.append(pred)

        return predictions
    
    def compute_accuracy(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute Top-1 accuracy.
        
        Args:
            predictions: List of predicted class indices
            labels: List of ground truth labels
            
        Returns:
            Dictionary with accuracy metrics
        """
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
    """
    ImageNet Query Sample Library for MLPerf LoadGen.
    
    This class implements the MLPerf QSL interface.
    """
    
    def __init__(
        self,
        data_path: str,
        val_map_path: Optional[str] = None,
        preprocessing: Optional[PreprocessingConfig] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 1024,
        **kwargs
    ):
        """
        Initialize ImageNet QSL.
        
        Args:
            data_path: Path to ImageNet validation images
            val_map_path: Path to validation map file
            preprocessing: Preprocessing configuration
            count: Total number of samples
            performance_sample_count: Number of samples for performance testing
        """
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
        """
        Load samples into memory.
        
        Args:
            sample_list: List of sample indices to load
        """
        if not self._dataset.is_loaded:
            self._dataset.load()
        
        logger.debug(f"Loading {len(sample_list)} query samples...")
        
        for sample_id in sample_list:
            if sample_id not in self._loaded_samples:
                data, _ = self._dataset.get_sample(sample_id)
                self._loaded_samples[sample_id] = data
    
    def unload_query_samples(self, sample_list: List[int]) -> None:
        """
        Unload samples from memory.
        
        Args:
            sample_list: List of sample indices to unload
        """
        for sample_id in sample_list:
            if sample_id in self._loaded_samples:
                del self._loaded_samples[sample_id]
    
    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        """
        Get features for a sample.
        
        Args:
            sample_id: Sample index
            
        Returns:
            Dictionary with input features
        """
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
