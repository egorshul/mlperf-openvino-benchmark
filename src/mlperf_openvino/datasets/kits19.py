"""KiTS19 dataset for 3D-UNet medical image segmentation benchmark.

Follows the MLCommons reference implementation:
https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# KiTS19 constants (MLPerf reference)
NUM_CLASSES = 3  # background, kidney, tumor
PATCH_SIZE = (128, 128, 128)
OVERLAP_FACTOR = 0.5

# Fixed preprocessing constants from MLPerf nnU-Net reference (global_vars.py).
# These were precomputed from foreground voxel statistics across the training set.
TARGET_SPACING = (1.6, 1.2, 1.2)  # target voxel spacing in mm (z, y, x)
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
MEAN_VAL = 101.0
STDDEV_VAL = 76.9
PADDING_VAL = -2.2  # (MIN_CLIP_VAL - MEAN_VAL) / STDDEV_VAL

# Dice score smoothing (MLPerf reference: accuracy_kits.py)
DICE_SMOOTH_NR = 1e-6
DICE_SMOOTH_DR = 1e-6


def _get_gaussian_importance_map(patch_size: Tuple[int, ...]) -> np.ndarray:
    """Create Gaussian importance map matching MLPerf reference (inference_utils.py).

    The reference builds a 3D Gaussian via sequential outer products of a 1D
    scipy.signal.gaussian, then applies a cube root transformation to flatten
    the distribution before normalizing.
    """
    try:
        from scipy.signal import gaussian as scipy_gaussian
    except ImportError:
        raise ImportError(
            "scipy is required for Gaussian kernel computation. "
            "Install with: pip install scipy"
        )

    n = patch_size[0]  # 128
    sigma = 0.125 * n  # 16.0

    g1 = scipy_gaussian(n, std=sigma)
    g2 = np.outer(g1, g1)
    g3 = np.outer(g2.flatten(), g1).reshape(n, n, n)

    # Cube root transformation (matches MLPerf reference)
    g3 = g3 ** (1.0 / 3.0)
    g3 /= g3.max()

    return g3.astype(np.float32)


def _compute_sliding_window_positions(
    volume_shape: Tuple[int, ...],
    patch_size: Tuple[int, ...],
    overlap_factor: float = 0.5,
) -> List[Tuple[int, ...]]:
    """Compute patch start positions (MLPerf reference: inference_utils.py).

    Uses step_size = patch_size * (1 - overlap_factor), matching the reference
    get_slice_for_sliding_window function exactly.
    """
    dim = len(patch_size)
    strides = [int(patch_size[i] * (1 - overlap_factor)) for i in range(dim)]

    # Number of positions per dimension (reference formula)
    size = [
        (volume_shape[i] - patch_size[i]) // strides[i] + 1
        if volume_shape[i] >= patch_size[i] else 1
        for i in range(dim)
    ]

    positions = []
    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                positions.append((i, j, k))

    return positions


def compute_dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_id: int,
) -> float:
    """Compute Dice score for a single class (MLPerf reference: accuracy_kits.py).

    Dice = (2 * intersection + smooth_nr) / (pred_sum + gt_sum + smooth_dr)
    """
    pred_mask = (prediction == class_id).astype(np.float64)
    gt_mask = (ground_truth == class_id).astype(np.float64)

    intersection = np.sum(pred_mask * gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    return float(
        (2.0 * intersection + DICE_SMOOTH_NR)
        / (pred_sum + gt_sum + DICE_SMOOTH_DR)
    )


class KiTS19Dataset(BaseDataset):
    """KiTS19 dataset for 3D medical image segmentation.

    Supports three data formats (checked in order):

    1. MLPerf reference preprocessed pickle format:
        data_path/
            preprocessed_files.pkl   # {"file_list": ["case_00XXX", ...]}
            case_00000.pkl
            case_00001.pkl
            ...

    2. Raw NIfTI format (will apply resampling + normalization):
        data_path/
            case_00000/
                imaging.nii.gz
                segmentation.nii.gz
            ...

    3. Preprocessed numpy format (already resampled + normalized):
        data_path/
            case_00000.npy
            case_00000_seg.npy
            ...
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        patch_size: Tuple[int, ...] = PATCH_SIZE,
        overlap_factor: float = OVERLAP_FACTOR,
        **kwargs,
    ):
        super().__init__(data_path, count, **kwargs)

        self.patch_size = patch_size
        self.overlap_factor = overlap_factor

        self._volumes: Dict[int, np.ndarray] = {}
        self._segmentations: Dict[int, np.ndarray] = {}
        self._case_names: List[str] = []
        self._data_format: Optional[str] = None  # "pickle", "nifti", "numpy"
        self._gaussian_map: Optional[np.ndarray] = None

    def load(self) -> None:
        """Load dataset metadata (case list)."""
        if self._loaded:
            return

        data_path = Path(self.data_path)

        # 1. Try MLPerf reference pickle format
        pkl_meta = data_path / "preprocessed_files.pkl"
        if pkl_meta.exists():
            self._load_pickle_metadata(data_path, pkl_meta)
            self._data_format = "pickle"
        else:
            # 2. Try NIfTI directory structure
            nifti_cases = sorted(
                d for d in data_path.iterdir()
                if d.is_dir() and d.name.startswith("case_")
            ) if data_path.exists() else []

            if nifti_cases:
                self._load_nifti_metadata(nifti_cases)
                self._data_format = "nifti"
            else:
                # 3. Try preprocessed numpy format
                self._load_numpy_metadata(data_path)
                self._data_format = "numpy"

        if self.count is not None:
            self._items = self._items[:self.count]
            self._case_names = self._case_names[:self.count]
            self._labels = self._labels[:self.count]

        logger.info(
            f"Found {len(self._items)} KiTS19 cases (format: {self._data_format})"
        )
        self._loaded = True

    def _load_pickle_metadata(self, data_path: Path, meta_path: Path) -> None:
        """Load MLPerf reference preprocessed pickle metadata."""
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        file_list = meta.get("file_list", [])
        for case_name in file_list:
            pkl_file = data_path / f"{case_name}.pkl"
            if pkl_file.exists():
                self._items.append(str(pkl_file))
                self._case_names.append(case_name)
                # Ground truth is stored within the pickle or as separate file
                self._labels.append(str(pkl_file))

    def _load_nifti_metadata(self, nifti_cases: List[Path]) -> None:
        """Load NIfTI directory metadata."""
        for case_dir in nifti_cases:
            imaging = case_dir / "imaging.nii.gz"
            if imaging.exists():
                self._items.append(str(case_dir))
                self._case_names.append(case_dir.name)
                self._labels.append(str(case_dir / "segmentation.nii.gz"))

    def _load_numpy_metadata(self, data_path: Path) -> None:
        """Load preprocessed numpy format metadata."""
        npy_files = sorted(data_path.glob("case_*[0-9].npy"))
        for npy_file in npy_files:
            self._items.append(str(npy_file))
            self._case_names.append(npy_file.stem)
            seg_file = npy_file.parent / f"{npy_file.stem}_seg.npy"
            self._labels.append(str(seg_file))

    def _load_nifti_file(self, path: str) -> Tuple[np.ndarray, Tuple[float, ...]]:
        """Load a NIfTI file and return (data, voxel_spacing).

        Spacing extracted from header pixdim, matching MLPerf reference.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for loading NIfTI files. "
                "Install with: pip install nibabel"
            )
        nii = nib.load(path)
        data = np.asarray(nii.dataobj, dtype=np.float32)
        spacing = tuple(float(s) for s in nii.header["pixdim"][1:4])
        return data, spacing

    def _resample_volume(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, ...],
        order: int = 1,
    ) -> np.ndarray:
        """Resample volume to target spacing [1.6, 1.2, 1.2] mm.

        Matches MLPerf reference preprocess.py using scipy.ndimage.zoom.
        order=1 (linear) for images, order=0 (nearest) for labels.
        """
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy is required for volume resampling. "
                "Install with: pip install scipy"
            )

        zoom_factors = np.array(spacing) / np.array(TARGET_SPACING)

        if np.allclose(zoom_factors, 1.0, atol=1e-3):
            return volume

        # Match reference: mode="constant", cval=min value, grid_mode=False
        cval = float(volume.min())
        resampled = zoom(
            volume, zoom_factors, order=order,
            mode="constant", cval=cval, grid_mode=False,
        )
        return resampled.astype(volume.dtype)

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess a 3D volume following MLPerf nnU-Net reference (global_vars.py).

        Uses FIXED dataset-level statistics (not per-volume):
        1. Clip to [-79.0, 304.0]
        2. Z-score normalize with mean=101.0, std=76.9
        """
        volume = np.clip(volume, MIN_CLIP_VAL, MAX_CLIP_VAL)
        volume = (volume - MEAN_VAL) / STDDEV_VAL
        return volume.astype(np.float32)

    def _pad_to_min_shape(
        self, volume: np.ndarray, label: Optional[np.ndarray] = None,
    ) -> Any:
        """Pad volume (and label) to at least patch_size using edge mode.

        Matches MLPerf reference preprocess.py pad_to_min_shape.
        """
        shape = volume.shape
        bounds = [max(0, self.patch_size[i] - shape[i]) for i in range(3)]

        if all(b == 0 for b in bounds):
            return (volume, label) if label is not None else volume

        paddings = [
            (bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)
        ]
        volume = np.pad(volume, paddings, mode="edge")
        if label is not None:
            label = np.pad(label, paddings, mode="edge")
            return volume, label
        return volume

    def _adjust_shape_for_sliding_window(
        self, volume: np.ndarray, label: Optional[np.ndarray] = None,
    ) -> Any:
        """Crop small remainders and pad to stride-aligned dimensions.

        Matches MLPerf reference preprocess.py adjust_shape_for_sliding_window:
        1. If dimension remainder < stride/2, crop symmetrically.
        2. Pad with constant value to make divisible by stride and >= patch_size.
           Image uses PADDING_VAL (-2.2), label uses 0.
        """
        shape = list(volume.shape)
        strides = [
            int(self.patch_size[i] * (1 - self.overlap_factor))
            for i in range(3)
        ]

        # Crop small remainders (< stride/2)
        remainders = [shape[i] % strides[i] for i in range(3)]
        crop = [
            remainders[i] if remainders[i] < strides[i] // 2 else 0
            for i in range(3)
        ]

        if any(c > 0 for c in crop):
            slices = tuple(
                slice(crop[i] // 2, shape[i] - (crop[i] - crop[i] // 2))
                for i in range(3)
            )
            volume = volume[slices]
            if label is not None:
                label = label[slices]

        # Pad to make divisible by stride and >= patch_size
        shape = list(volume.shape)
        pad_amounts = [
            (strides[i] - shape[i] % strides[i]) % strides[i]
            for i in range(3)
        ]
        pad_amounts = [
            pad_amounts[i]
            if (shape[i] + pad_amounts[i]) >= self.patch_size[i]
            else pad_amounts[i] + strides[i]
            for i in range(3)
        ]

        if any(p > 0 for p in pad_amounts):
            paddings = [
                (pad_amounts[i] // 2, pad_amounts[i] - pad_amounts[i] // 2)
                for i in range(3)
            ]
            volume = np.pad(
                volume, paddings, mode="constant", constant_values=PADDING_VAL,
            )
            if label is not None:
                label = np.pad(
                    label, paddings, mode="constant", constant_values=0,
                )

        return (volume, label) if label is not None else volume

    def _preprocess_nifti_case(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess a NIfTI case matching full MLPerf reference pipeline.

        Steps (from reference preprocess.py):
        1. Load NIfTI volume and label
        2. Resample to target spacing [1.6, 1.2, 1.2] mm
        3. Normalize intensity (clip + z-score)
        4. Pad to minimum shape (128^3) using edge mode
        5. Adjust shape for sliding window (crop + constant pad)
        """
        case_dir = Path(self._items[index])

        # Step 1: Load
        volume, spacing = self._load_nifti_file(str(case_dir / "imaging.nii.gz"))

        seg = None
        seg_path = Path(self._labels[index])
        if seg_path.exists():
            seg_data, _ = self._load_nifti_file(str(seg_path))
            seg = seg_data.astype(np.uint8)

        # Step 2: Resample
        volume = self._resample_volume(volume, spacing, order=1)
        if seg is not None:
            seg_float = seg.astype(np.float32)
            seg_float = self._resample_volume(seg_float, spacing, order=0)
            seg = seg_float.astype(np.uint8)

        # Step 3: Normalize
        volume = self._preprocess_volume(volume)

        # Step 4: Pad to min shape
        if seg is not None:
            volume, seg = self._pad_to_min_shape(volume, seg)
        else:
            volume = self._pad_to_min_shape(volume)

        # Step 5: Adjust for sliding window
        if seg is not None:
            volume, seg = self._adjust_shape_for_sliding_window(volume, seg)
        else:
            volume = self._adjust_shape_for_sliding_window(volume)

        return volume.astype(np.float32), seg

    def _load_volume(self, index: int) -> np.ndarray:
        """Load and preprocess a volume by index."""
        path = self._items[index]

        if self._data_format == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
            # MLPerf reference pickle: data is a list/tuple, volume at index 0
            volume = np.array(data[0], dtype=np.float32)
            # Pickle data is already preprocessed (resampled + normalized + padded)
            return volume

        elif self._data_format == "numpy":
            # Numpy format assumed already preprocessed
            return np.load(path).astype(np.float32)

        else:
            # NIfTI: full preprocessing pipeline (both volume and label together)
            volume, seg = self._preprocess_nifti_case(index)
            # Cache segmentation too since it was preprocessed together
            if seg is not None:
                self._segmentations[index] = seg
            return volume

    def _load_segmentation(self, index: int) -> np.ndarray:
        """Load ground truth segmentation by index."""
        path = self._labels[index]

        if self._data_format == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
            # MLPerf reference pickle: segmentation at index 1
            return np.array(data[1], dtype=np.uint8)

        elif self._data_format == "numpy":
            return np.load(path).astype(np.uint8)

        else:
            # NIfTI: preprocessed together with volume
            # If not yet cached, trigger full preprocessing
            if index not in self._segmentations:
                volume, seg = self._preprocess_nifti_case(index)
                self._volumes[index] = volume
                if seg is not None:
                    self._segmentations[index] = seg
                else:
                    raise FileNotFoundError(
                        f"Segmentation not found for case {index}: {path}"
                    )
            return self._segmentations[index]

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
        """Get Gaussian importance map (MLPerf reference: scipy + cube root)."""
        if self._gaussian_map is None:
            self._gaussian_map = _get_gaussian_importance_map(self.patch_size)
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
            return [
                np.argmax(results[i], axis=0).astype(np.uint8)
                for i in range(results.shape[0])
            ]
        elif results.ndim == 4:
            # (classes, D, H, W) -> (D, H, W)
            return [np.argmax(results, axis=0).astype(np.uint8)]
        return [results.astype(np.uint8)]

    def compute_accuracy(
        self,
        predictions: List[np.ndarray],
        labels: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute mean Dice score (MLPerf 3D-UNet metric).

        Per-case: mean of kidney Dice and tumor Dice.
        Overall: mean across all cases. NaN treated as 0 per reference.
        """
        case_dices = []

        for pred, gt in zip(predictions, labels):
            kidney_dice = compute_dice_score(pred, gt, class_id=1)
            tumor_dice = compute_dice_score(pred, gt, class_id=2)
            case_mean = (kidney_dice + tumor_dice) / 2.0
            case_dices.append({
                "kidney": kidney_dice,
                "tumor": tumor_dice,
                "mean": case_mean,
            })

        # Overall means (NaN -> 0 per reference)
        kidney_dices = [c["kidney"] for c in case_dices]
        tumor_dices = [c["tumor"] for c in case_dices]
        mean_dices = [c["mean"] for c in case_dices]

        overall_mean = float(np.nanmean(mean_dices)) if mean_dices else 0.0
        overall_kidney = float(np.nanmean(kidney_dices)) if kidney_dices else 0.0
        overall_tumor = float(np.nanmean(tumor_dices)) if tumor_dices else 0.0

        return {
            "mean_dice": overall_mean,
            "kidney_dice": overall_kidney,
            "tumor_dice": overall_tumor,
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
