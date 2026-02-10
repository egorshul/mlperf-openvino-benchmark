"""KiTS 2019 dataset for 3D UNET medical image segmentation benchmark.

MLPerf Inference v5.1 - Kidney Tumor Segmentation (KiTS 2019).
Implements sliding window inference with Gaussian importance weighting
per MLCommons reference implementation.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# MLPerf KiTS19 preprocessing constants
TARGET_SPACING = (1.6, 1.2, 1.2)  # Target voxel spacing in mm (z, y, x)
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
NORM_MEAN = 101.0
NORM_STD = 76.9
ROI_SHAPE = (128, 128, 128)
SLIDE_OVERLAP_FACTOR = 0.5
PAD_VALUE = -2.2  # (MIN_CLIP_VAL - NORM_MEAN) / NORM_STD ≈ -2.34, but MLPerf uses -2.2


def get_gaussian_importance_map(roi_shape: Tuple[int, ...], sigma_scale: float = 0.125) -> np.ndarray:
    """Create 3D Gaussian importance map for sliding window aggregation.

    Per MLCommons reference: sigma = sigma_scale * roi_shape[0].
    """
    tmp = np.zeros(roi_shape, dtype=np.float32)
    center = [i // 2 for i in roi_shape]
    sigmas = [i * sigma_scale for i in roi_shape]
    tmp[tuple(center)] = 1.0

    try:
        from scipy.ndimage import gaussian_filter
        gaussian = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    except ImportError:
        # Fallback: create separable Gaussian
        axes = [np.arange(s, dtype=np.float32) for s in roi_shape]
        gaussian = np.ones(roi_shape, dtype=np.float32)
        for ax_idx, (ax, c, sig) in enumerate(zip(axes, center, sigmas)):
            shape = [1] * len(roi_shape)
            shape[ax_idx] = len(ax)
            g = np.exp(-0.5 * ((ax - c) / sig) ** 2).reshape(shape)
            gaussian *= g
        gaussian = gaussian.astype(np.float32)

    # Cube root transform per MLCommons reference
    gaussian = np.power(gaussian, 1.0 / 3.0)
    gaussian[gaussian == 0] = np.min(gaussian[gaussian != 0])
    gaussian /= gaussian.max()
    return gaussian.astype(np.float32)


def compute_sliding_window_positions(
    image_shape: Tuple[int, ...],
    roi_shape: Tuple[int, ...] = ROI_SHAPE,
    overlap: float = SLIDE_OVERLAP_FACTOR,
) -> List[Tuple[slice, ...]]:
    """Compute sliding window positions with given overlap factor."""
    positions = []
    steps = [max(1, int(r * (1.0 - overlap))) for r in roi_shape]

    for d_start in range(0, max(1, image_shape[0] - roi_shape[0] + 1), steps[0]):
        d_end = min(d_start + roi_shape[0], image_shape[0])
        d_start = d_end - roi_shape[0]

        for h_start in range(0, max(1, image_shape[1] - roi_shape[1] + 1), steps[1]):
            h_end = min(h_start + roi_shape[1], image_shape[1])
            h_start = h_end - roi_shape[1]

            for w_start in range(0, max(1, image_shape[2] - roi_shape[2] + 1), steps[2]):
                w_end = min(w_start + roi_shape[2], image_shape[2])
                w_start = w_end - roi_shape[2]

                slices = (
                    slice(d_start, d_end),
                    slice(h_start, h_end),
                    slice(w_start, w_end),
                )
                positions.append(slices)

    return positions


def pad_to_min_shape(data: np.ndarray, min_shape: Tuple[int, ...], pad_value: float = PAD_VALUE) -> np.ndarray:
    """Pad 3D volume to minimum shape, with dimensions divisible by 64."""
    padded_shape = []
    for i, (s, m) in enumerate(zip(data.shape, min_shape)):
        target = max(s, m)
        # Make divisible by 64 for sliding window compatibility
        target = int(np.ceil(target / 64.0)) * 64
        padded_shape.append(target)

    if tuple(padded_shape) == data.shape:
        return data

    padded = np.full(padded_shape, pad_value, dtype=data.dtype)
    padded[:data.shape[0], :data.shape[1], :data.shape[2]] = data
    return padded


def preprocess_volume(image: np.ndarray) -> np.ndarray:
    """Preprocess a single 3D volume per MLCommons KiTS19 specification.

    Steps: clip -> z-score normalize -> pad.
    Resampling is expected to be done beforehand (during dataset download).
    """
    image = np.clip(image, MIN_CLIP_VAL, MAX_CLIP_VAL)
    image = (image - NORM_MEAN) / NORM_STD
    image = pad_to_min_shape(image, ROI_SHAPE)
    return image.astype(np.float32)


def dice_score(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Compute Dice score for a specific class.

    DICE = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    """
    eps = 1e-6
    pred_mask = (prediction == class_id).astype(np.float32)
    target_mask = (target == class_id).astype(np.float32)
    intersection = np.sum(pred_mask * target_mask)
    return float((2.0 * intersection + eps) / (np.sum(pred_mask) + np.sum(target_mask) + eps))


class KiTS19Dataset(BaseDataset):
    """KiTS 2019 dataset for 3D UNET segmentation.

    Expects preprocessed data in pickle format (per MLCommons reference)
    or raw NIfTI volumes.
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.count = count
        self.samples: Dict[int, Dict[str, Any]] = {}
        self.case_ids: List[str] = []
        self.labels: Dict[int, np.ndarray] = {}
        self._total_count = 0

    @property
    def total_count(self) -> int:
        return self._total_count

    def load(self) -> None:
        """Load dataset, discovering available preprocessed cases."""
        preprocessed_dir = self.data_path / "preprocessed"
        raw_dir = self.data_path / "raw"

        if preprocessed_dir.exists():
            self._load_preprocessed(preprocessed_dir)
        elif raw_dir.exists():
            self._load_nifti(raw_dir)
        elif self.data_path.exists():
            # Try loading directly from data_path
            pkl_files = sorted(self.data_path.glob("case_*/*.pkl")) or \
                        sorted(self.data_path.glob("*.pkl"))
            nii_files = sorted(self.data_path.glob("case_*/imaging.nii.gz"))

            if pkl_files:
                self._load_preprocessed(self.data_path)
            elif nii_files:
                self._load_nifti(self.data_path)
            else:
                raise FileNotFoundError(
                    f"No preprocessed (.pkl) or NIfTI (.nii.gz) files found in {self.data_path}. "
                    f"Run: mlperf-ov download-dataset --dataset kits19"
                )
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        if self.count and self.count < self._total_count:
            self._total_count = self.count
            self.case_ids = self.case_ids[:self.count]

        logger.info(f"KiTS19: Loaded {self._total_count} cases from {self.data_path}")

    def _load_preprocessed(self, data_dir: Path) -> None:
        """Load preprocessed pickle files (MLCommons format)."""
        # Look for case directories or direct pickle files
        case_dirs = sorted(data_dir.glob("case_*"))
        if case_dirs:
            for case_dir in case_dirs:
                pkl_file = case_dir / "data.pkl"
                if not pkl_file.exists():
                    # Try any .pkl file in the case directory
                    pkl_files = list(case_dir.glob("*.pkl"))
                    if pkl_files:
                        pkl_file = pkl_files[0]
                    else:
                        continue
                self.case_ids.append(case_dir.name)
        else:
            # Direct pickle files
            pkl_files = sorted(data_dir.glob("*.pkl"))
            for pkl_file in pkl_files:
                self.case_ids.append(pkl_file.stem)

        self._total_count = len(self.case_ids)

    def _load_nifti(self, data_dir: Path) -> None:
        """Load raw NIfTI volumes."""
        case_dirs = sorted(data_dir.glob("case_*"))
        for case_dir in case_dirs:
            imaging_file = case_dir / "imaging.nii.gz"
            if imaging_file.exists():
                self.case_ids.append(case_dir.name)

        self._total_count = len(self.case_ids)

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a preprocessed sample by index.

        Returns dict with 'data' (preprocessed volume) and 'original_shape'.
        """
        if index in self.samples:
            return self.samples[index]

        case_id = self.case_ids[index]
        return self._load_case(case_id, index)

    def _load_case(self, case_id: str, index: int) -> Dict[str, Any]:
        """Load and preprocess a single case."""
        preprocessed_dir = self.data_path / "preprocessed"
        raw_dir = self.data_path / "raw"

        # Try preprocessed first
        for base_dir in [preprocessed_dir, self.data_path]:
            pkl_path = base_dir / case_id / "data.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                sample = {
                    "data": data["data"] if isinstance(data, dict) and "data" in data else data,
                    "case_id": case_id,
                    "original_shape": data.get("original_shape", None) if isinstance(data, dict) else None,
                }
                self.samples[index] = sample

                # Load label if available
                label_path = base_dir / case_id / "label.pkl"
                if label_path.exists():
                    with open(label_path, "rb") as f:
                        self.labels[index] = pickle.load(f)

                return sample

        # Try NIfTI
        for base_dir in [raw_dir, self.data_path]:
            nii_path = base_dir / case_id / "imaging.nii.gz"
            if nii_path.exists():
                return self._load_nifti_case(nii_path, case_id, index, base_dir)

        raise FileNotFoundError(f"Case {case_id} not found in {self.data_path}")

    def _load_nifti_case(self, nii_path: Path, case_id: str, index: int, base_dir: Path) -> Dict[str, Any]:
        """Load and preprocess a NIfTI case."""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required for loading NIfTI files: pip install nibabel")

        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.float32)

        # Resample to target spacing
        original_spacing = img.header.get_zooms()[:3]
        data = self._resample_volume(data, original_spacing, TARGET_SPACING)

        original_shape = data.shape
        data = preprocess_volume(data)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        data = data[np.newaxis, ...]

        sample = {
            "data": data,
            "case_id": case_id,
            "original_shape": original_shape,
        }
        self.samples[index] = sample

        # Load segmentation label if available
        label_path = base_dir / case_id / "segmentation.nii.gz"
        if label_path.exists():
            label_img = nib.load(str(label_path))
            label = label_img.get_fdata().astype(np.int8)
            label = self._resample_volume(label, original_spacing, TARGET_SPACING, order=0)
            self.labels[index] = label

        return sample

    @staticmethod
    def _resample_volume(
        data: np.ndarray,
        original_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...],
        order: int = 1,
    ) -> np.ndarray:
        """Resample a 3D volume to target voxel spacing."""
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError("scipy is required for resampling: pip install scipy")

        zoom_factors = tuple(o / t for o, t in zip(original_spacing, target_spacing))
        if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
            return data
        return zoom(data, zoom_factors, order=order).astype(data.dtype)

    def get_label(self, index: int) -> Optional[np.ndarray]:
        """Get ground truth segmentation label for a sample."""
        return self.labels.get(index, None)

    def postprocess(self, result: np.ndarray, sample_indices: List[int]) -> List[np.ndarray]:
        """Postprocess model output: argmax over class dimension."""
        # result shape: (num_classes, D, H, W) -> (D, H, W) via argmax
        if result.ndim == 4:
            return [np.argmax(result, axis=0).astype(np.int8)]
        elif result.ndim == 5:
            # Batched: (B, C, D, H, W)
            return [np.argmax(result[i], axis=0).astype(np.int8) for i in range(result.shape[0])]
        return [result.astype(np.int8)]

    def compute_accuracy(self, predictions: List[np.ndarray], labels: List[np.ndarray]) -> Dict[str, Any]:
        """Compute mean Dice score (kidney + tumor)."""
        kidney_dices = []
        tumor_dices = []

        for pred, label in zip(predictions, labels):
            if label is None:
                continue
            # Class 1 = kidney, Class 2 = tumor
            kidney_dices.append(dice_score(pred, label, class_id=1))
            tumor_dices.append(dice_score(pred, label, class_id=2))

        if not kidney_dices:
            return {"mean_dice": 0.0, "kidney_dice": 0.0, "tumor_dice": 0.0, "num_samples": 0}

        mean_kidney = float(np.mean(kidney_dices))
        mean_tumor = float(np.mean(tumor_dices))
        mean_dice = (mean_kidney + mean_tumor) / 2.0

        return {
            "mean_dice": mean_dice,
            "kidney_dice": mean_kidney,
            "tumor_dice": mean_tumor,
            "num_samples": len(kidney_dices),
        }


class KiTS19QSL(QuerySampleLibrary):
    """Query Sample Library for KiTS 2019 dataset."""

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        performance_sample_count: int = 42,
    ):
        self.dataset = KiTS19Dataset(data_path=data_path, count=count)
        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, Dict[str, Any]] = {}

    def load(self) -> None:
        """Load dataset metadata."""
        self.dataset.load()

    def load_query_samples(self, sample_list: List[int]) -> None:
        """Load samples into memory."""
        for idx in sample_list:
            if idx not in self._loaded_samples:
                self._loaded_samples[idx] = self.dataset.get_sample(idx)

    def unload_query_samples(self, sample_list: List[int]) -> None:
        """Unload samples from memory."""
        for idx in sample_list:
            self._loaded_samples.pop(idx, None)
            self.dataset.samples.pop(idx, None)

    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        """Get preprocessed features for a sample.

        Returns dict with 'input': preprocessed 3D volume as (1, C, D, H, W) tensor.
        """
        sample = self._loaded_samples.get(sample_id)
        if sample is None:
            sample = self.dataset.get_sample(sample_id)
            self._loaded_samples[sample_id] = sample

        data = sample["data"]
        # Ensure shape is (1, C, D, H, W)
        if data.ndim == 3:
            data = data[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
        elif data.ndim == 4:
            data = data[np.newaxis, ...]  # (1, C, D, H, W)

        return {"input": data.astype(np.float32)}

    def get_label(self, sample_id: int) -> Optional[np.ndarray]:
        """Get ground truth label."""
        return self.dataset.get_label(sample_id)

    @property
    def total_sample_count(self) -> int:
        return self.dataset.total_count

    @property
    def performance_sample_count(self) -> int:
        # MLPerf 3D UNET: use all samples (42) for performance
        return min(self._performance_sample_count, self.dataset.total_count)
