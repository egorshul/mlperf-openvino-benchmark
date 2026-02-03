"""KiTS19 dataset for 3D-UNet medical image segmentation benchmark."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# KiTS19 constants (MLPerf reference)
NUM_CLASSES = 3  # background, kidney, tumor
PATCH_SIZE = (128, 128, 128)
OVERLAP_FACTOR = 0.5
GAUSSIAN_SIGMA_SCALE = 0.125

# Preprocessing constants (MLPerf nnU-Net reference)
CLIP_LOW_PERCENTILE = 0.5
CLIP_HIGH_PERCENTILE = 99.5


def _get_gaussian_importance_map(
    patch_size: Tuple[int, ...],
    sigma_scale: float = 0.125,
) -> np.ndarray:
    """Create Gaussian importance map for sliding window aggregation.

    This matches the nnU-Net reference implementation used by MLPerf.
    """
    center = [(s - 1) / 2.0 for s in patch_size]
    sigma = [s * sigma_scale for s in patch_size]

    grid = np.ogrid[tuple(slice(0, s) for s in patch_size)]
    gaussian = np.ones(patch_size, dtype=np.float32)

    for i, (g, c, s) in enumerate(zip(grid, center, sigma)):
        gaussian *= np.exp(-0.5 * ((g - c) / s) ** 2)

    gaussian /= gaussian.max()
    gaussian = np.clip(gaussian, a_min=1e-7, a_max=None)
    return gaussian


def _compute_sliding_window_positions(
    volume_shape: Tuple[int, ...],
    patch_size: Tuple[int, ...],
    overlap_factor: float = 0.5,
) -> List[Tuple[int, ...]]:
    """Compute patch start positions for sliding window inference.

    Uses step_size = patch_size * (1 - overlap_factor), matching MLPerf reference.
    """
    positions = []
    steps = [max(1, int(p * (1 - overlap_factor))) for p in patch_size]

    ranges = []
    for dim_size, p_size, step in zip(volume_shape, patch_size, steps):
        starts = list(range(0, max(dim_size - p_size + 1, 1), step))
        # Ensure last patch covers the end
        if starts[-1] + p_size < dim_size:
            starts.append(dim_size - p_size)
        ranges.append(starts)

    # Generate all combinations
    for d in ranges[0]:
        for h in ranges[1]:
            for w in ranges[2]:
                positions.append((d, h, w))

    return positions


def compute_dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_id: int,
    smooth: float = 1e-5,
) -> float:
    """Compute Dice score for a single class."""
    pred_mask = (prediction == class_id).astype(np.float32)
    gt_mask = (ground_truth == class_id).astype(np.float32)

    intersection = np.sum(pred_mask * gt_mask)
    denominator = np.sum(pred_mask) + np.sum(gt_mask)

    if denominator < smooth:
        return 1.0 if np.sum(gt_mask) == 0 else 0.0

    return float(2.0 * intersection / (denominator + smooth))


class KiTS19Dataset(BaseDataset):
    """KiTS19 dataset for 3D medical image segmentation.

    Expected directory structure:
        data_path/
            case_00000/
                imaging.nii.gz
                segmentation.nii.gz
            case_00001/
                imaging.nii.gz
                segmentation.nii.gz
            ...
        OR (preprocessed numpy format):
            case_00000.npy (imaging)
            case_00000_seg.npy (segmentation)
            ...
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        patch_size: Tuple[int, ...] = PATCH_SIZE,
        overlap_factor: float = OVERLAP_FACTOR,
        gaussian_sigma_scale: float = GAUSSIAN_SIGMA_SCALE,
        **kwargs,
    ):
        super().__init__(data_path, count, **kwargs)

        self.patch_size = patch_size
        self.overlap_factor = overlap_factor
        self.gaussian_sigma_scale = gaussian_sigma_scale

        self._volumes: Dict[int, np.ndarray] = {}
        self._segmentations: Dict[int, np.ndarray] = {}
        self._case_names: List[str] = []
        self._gaussian_map: Optional[np.ndarray] = None

    def load(self) -> None:
        """Load dataset metadata (case list)."""
        if self._loaded:
            return

        data_path = Path(self.data_path)

        # Try NIfTI directory structure first
        nifti_cases = sorted(
            d for d in data_path.iterdir()
            if d.is_dir() and d.name.startswith("case_")
        ) if data_path.exists() else []

        if nifti_cases:
            for case_dir in nifti_cases:
                imaging = case_dir / "imaging.nii.gz"
                if imaging.exists():
                    self._items.append(str(case_dir))
                    self._case_names.append(case_dir.name)
                    self._labels.append(str(case_dir / "segmentation.nii.gz"))
        else:
            # Try preprocessed numpy format
            npy_files = sorted(data_path.glob("case_*[0-9].npy"))
            for npy_file in npy_files:
                self._items.append(str(npy_file))
                self._case_names.append(npy_file.stem)
                seg_file = npy_file.parent / f"{npy_file.stem}_seg.npy"
                self._labels.append(str(seg_file))

        if self.count is not None:
            self._items = self._items[:self.count]
            self._case_names = self._case_names[:self.count]
            self._labels = self._labels[:self.count]

        logger.info(f"Found {len(self._items)} KiTS19 cases")
        self._loaded = True

    def _load_nifti(self, path: str) -> np.ndarray:
        """Load a NIfTI file and return numpy array."""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for loading NIfTI files. "
                "Install with: pip install nibabel"
            )
        nii = nib.load(path)
        return np.asarray(nii.dataobj, dtype=np.float32)

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess a 3D volume following MLPerf nnU-Net reference.

        1. Clip to [p0.5, p99.5] percentiles
        2. Z-score normalization (mean=0, std=1)
        """
        # Clip to percentile range
        low = np.percentile(volume, CLIP_LOW_PERCENTILE)
        high = np.percentile(volume, CLIP_HIGH_PERCENTILE)
        volume = np.clip(volume, low, high)

        # Z-score normalization
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume = (volume - mean) / std
        else:
            volume = volume - mean

        return volume.astype(np.float32)

    def _load_volume(self, index: int) -> np.ndarray:
        """Load and preprocess a volume by index."""
        path = self._items[index]

        if path.endswith(".npy"):
            volume = np.load(path).astype(np.float32)
        else:
            # NIfTI directory
            imaging_path = Path(path) / "imaging.nii.gz"
            volume = self._load_nifti(str(imaging_path))

        return self._preprocess_volume(volume)

    def _load_segmentation(self, index: int) -> np.ndarray:
        """Load ground truth segmentation by index."""
        path = self._labels[index]

        if path.endswith(".npy"):
            return np.load(path).astype(np.int64)
        else:
            seg = self._load_nifti(path)
            return seg.astype(np.int64)

    def get_volume(self, index: int) -> np.ndarray:
        """Get preprocessed volume (cached)."""
        if index not in self._volumes:
            self._volumes[index] = self._load_volume(index)
        return self._volumes[index]

    def get_segmentation(self, index: int) -> np.ndarray:
        """Get ground truth segmentation (cached)."""
        if index not in self._segmentations:
            self._segmentations[index] = self._load_segmentation(index)
        return self._segmentations[index]

    def get_gaussian_map(self) -> np.ndarray:
        """Get the Gaussian importance map for patch aggregation."""
        if self._gaussian_map is None:
            self._gaussian_map = _get_gaussian_importance_map(
                self.patch_size, self.gaussian_sigma_scale
            )
        return self._gaussian_map

    def get_sliding_window_positions(
        self, volume_shape: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        """Get sliding window patch positions for a volume."""
        return _compute_sliding_window_positions(
            volume_shape, self.patch_size, self.overlap_factor,
        )

    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a preprocessed volume and its segmentation."""
        if not self._loaded:
            self.load()
        volume = self.get_volume(index)
        seg = self.get_segmentation(index)
        return volume, seg

    def get_samples(
        self, indices: List[int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get multiple volumes and segmentations."""
        if not self._loaded:
            self.load()
        volumes = [self.get_volume(i) for i in indices]
        segs = [self.get_segmentation(i) for i in indices]
        return volumes, segs

    def postprocess(
        self,
        results: np.ndarray,
        indices: List[int],
    ) -> List[np.ndarray]:
        """Convert model output logits to class predictions (argmax)."""
        if results.ndim == 5:
            # (batch, classes, D, H, W) -> (batch, D, H, W)
            return [np.argmax(results[i], axis=0) for i in range(results.shape[0])]
        elif results.ndim == 4:
            # (classes, D, H, W) -> (D, H, W)
            return [np.argmax(results, axis=0)]
        return [results]

    def compute_accuracy(
        self,
        predictions: List[np.ndarray],
        labels: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute mean Dice score (MLPerf 3D-UNet metric).

        Computes Dice for kidney (class 1) and tumor (class 2),
        returns the mean as the primary metric.
        """
        kidney_dices = []
        tumor_dices = []

        for pred, gt in zip(predictions, labels):
            kidney_dice = compute_dice_score(pred, gt, class_id=1)
            tumor_dice = compute_dice_score(pred, gt, class_id=2)
            kidney_dices.append(kidney_dice)
            tumor_dices.append(tumor_dice)

        mean_kidney = float(np.mean(kidney_dices)) if kidney_dices else 0.0
        mean_tumor = float(np.mean(tumor_dices)) if tumor_dices else 0.0
        mean_dice = (mean_kidney + mean_tumor) / 2.0

        return {
            "mean_dice": mean_dice,
            "kidney_dice": mean_kidney,
            "tumor_dice": mean_tumor,
            "num_samples": len(predictions),
        }


class KiTS19QSL(QuerySampleLibrary):
    """KiTS19 Query Sample Library for MLPerf LoadGen."""

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        performance_sample_count: int = 42,
        patch_size: Tuple[int, ...] = PATCH_SIZE,
        overlap_factor: float = OVERLAP_FACTOR,
        **kwargs,
    ):
        self._dataset = KiTS19Dataset(
            data_path=data_path,
            count=count,
            patch_size=patch_size,
            overlap_factor=overlap_factor,
            **kwargs,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, np.ndarray] = {}

    def load(self) -> None:
        """Load the dataset."""
        self._dataset.load()

    def load_query_samples(self, sample_list: List[int]) -> None:
        """Load and preprocess volumes into memory."""
        if not self._dataset.is_loaded:
            self._dataset.load()

        to_load = [idx for idx in sample_list if idx not in self._loaded_samples]
        if not to_load:
            return

        import sys

        logger.debug(f"Loading {len(to_load)} KiTS19 volumes...")
        for i, idx in enumerate(to_load):
            volume = self._dataset.get_volume(idx)
            self._loaded_samples[idx] = volume
            if (i + 1) % 5 == 0 or (i + 1) == len(to_load):
                pct = (i + 1) / len(to_load) * 100
                sys.stderr.write(
                    f"\rLoading KiTS19 volumes: {i + 1}/{len(to_load)} ({pct:.1f}%)   "
                )
                sys.stderr.flush()

        if to_load:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def unload_query_samples(self, sample_list: List[int]) -> None:
        """Unload volumes from memory."""
        for sample_id in sample_list:
            self._loaded_samples.pop(sample_id, None)
            self._dataset._volumes.pop(sample_id, None)

    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        """Get preprocessed volume for a sample.

        Returns the full 3D volume. The SUT is responsible for
        sliding window patch extraction and aggregation.
        """
        if sample_id in self._loaded_samples:
            volume = self._loaded_samples[sample_id]
        else:
            volume = self._dataset.get_volume(sample_id)

        return {"volume": volume}

    def get_label(self, sample_id: int) -> np.ndarray:
        """Get ground truth segmentation for a sample."""
        return self._dataset.get_segmentation(sample_id)

    @property
    def total_sample_count(self) -> int:
        if not self._dataset.is_loaded:
            self._dataset.load()
        return self._dataset.sample_count

    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)

    @property
    def dataset(self) -> KiTS19Dataset:
        return self._dataset
