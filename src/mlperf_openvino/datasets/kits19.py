"""KiTS 2019 dataset for 3D UNET medical image segmentation benchmark.

MLPerf Inference v5.1 - Kidney Tumor Segmentation (KiTS 2019).
Implements sliding window inference with Gaussian importance weighting
per MLCommons reference implementation.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Official MLPerf inference case list (43 cases from meta/inference_cases.json)
MLPERF_INFERENCE_CASES = [
    "case_00000", "case_00003", "case_00005", "case_00006", "case_00012",
    "case_00024", "case_00034", "case_00041", "case_00044", "case_00049",
    "case_00052", "case_00056", "case_00061", "case_00065", "case_00066",
    "case_00070", "case_00076", "case_00078", "case_00080", "case_00084",
    "case_00086", "case_00087", "case_00092", "case_00111", "case_00112",
    "case_00125", "case_00128", "case_00138", "case_00157", "case_00160",
    "case_00161", "case_00162", "case_00169", "case_00171", "case_00176",
    "case_00185", "case_00187", "case_00189", "case_00198", "case_00203",
    "case_00206", "case_00207", "case_00400",
]


def get_gaussian_importance_map(roi_shape: Tuple[int, ...], sigma_scale: float = 0.125) -> np.ndarray:
    """Create 3D Gaussian importance map for sliding window aggregation.

    Matches MLCommons reference inference_utils.py gaussian_kernel() exactly:
    Uses scipy.signal.windows.gaussian with outer products and cube root normalization.
    """
    from scipy.signal.windows import gaussian

    n = roi_shape[0]
    std = sigma_scale * n
    gaussian1D = gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D).reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return gaussian3D.astype(np.float32)


def compute_sliding_window_positions(
    image_shape: Tuple[int, ...],
    roi_shape: Tuple[int, ...] = ROI_SHAPE,
    overlap: float = SLIDE_OVERLAP_FACTOR,
) -> List[Tuple[slice, ...]]:
    """Compute sliding window positions per MLCommons reference.

    Matches inference_utils.py get_slice_for_sliding_window() exactly.
    Assumes image dimensions are already adjusted for sliding window
    (divisible by stride via adjust_shape_for_sliding_window).
    """
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
    size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]

    positions = []
    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                slices = (
                    slice(i, i + roi_shape[0]),
                    slice(j, j + roi_shape[1]),
                    slice(k, k + roi_shape[2]),
                )
                positions.append(slices)

    return positions


def pad_to_min_shape(
    image: np.ndarray,
    label: np.ndarray = None,
) -> "Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]":
    """Pad 4D volumes (C,D,H,W) to minimum ROI shape using symmetric edge padding.

    Matches MLCommons reference preprocess.py pad_to_min_shape() exactly.
    """
    current_shape = image.shape[1:]  # spatial dims
    bounds = [max(0, ROI_SHAPE[i] - current_shape[i]) for i in range(3)]

    if all(b == 0 for b in bounds):
        if label is not None:
            return image, label
        return image

    paddings = [(0, 0)]  # channel dim
    paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)])

    image = np.pad(image, paddings, mode="edge")
    if label is not None:
        label = np.pad(label, paddings, mode="edge")
        return image, label
    return image


def adjust_shape_for_sliding_window(
    image: np.ndarray,
    label: np.ndarray = None,
    roi_shape: Tuple[int, ...] = ROI_SHAPE,
    overlap: float = SLIDE_OVERLAP_FACTOR,
    padding_val: float = PAD_VALUE,
) -> "Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]":
    """Adjust volume shape for sliding window inference per MLCommons reference.

    Matches preprocess.py adjust_shape_for_sliding_window() + constant_pad_volume().
    Two steps:
    1. Crop dimensions where remainder < stride/2
    2. Pad to make divisible by stride (symmetric, constant mode)

    Input/output arrays are 4D: (C, D, H, W).
    """
    image_shape = list(image.shape[1:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    # Step 1: Crop dimensions where remainder < stride/2
    bounds = [image_shape[i] % strides[i] for i in range(dim)]
    bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]

    image = image[
        ...,
        bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
        bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
        bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
    ]
    if label is not None:
        label = label[
            ...,
            bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
            bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
            bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
        ]

    # Step 2: Constant pad to make divisible by stride
    def _constant_pad(volume: np.ndarray, pad_val: float) -> np.ndarray:
        vol_shape = list(volume.shape[1:])
        pad_bounds = [(strides[i] - vol_shape[i] % strides[i]) % strides[i] for i in range(dim)]
        pad_bounds = [
            pad_bounds[i] if (vol_shape[i] + pad_bounds[i]) >= roi_shape[i]
            else pad_bounds[i] + strides[i]
            for i in range(dim)
        ]
        paddings = [
            (0, 0),  # channel dim
            (pad_bounds[0] // 2, pad_bounds[0] - pad_bounds[0] // 2),
            (pad_bounds[1] // 2, pad_bounds[1] - pad_bounds[1] // 2),
            (pad_bounds[2] // 2, pad_bounds[2] - pad_bounds[2] // 2),
        ]
        return np.pad(volume, paddings, mode="constant", constant_values=[pad_val])

    image = _constant_pad(image, padding_val)
    if label is not None:
        label = _constant_pad(label, 0)
        return image, label
    return image


def preprocess_volume(image: np.ndarray, label: np.ndarray = None) -> "Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]":
    """Preprocess a 4D volume (C,D,H,W) per MLCommons reference pipeline.

    Steps: clip → z-score normalize → pad_to_min_shape → adjust_shape_for_sliding_window.
    Resampling and channel dim expansion must be done beforehand.
    """
    image = np.clip(image, MIN_CLIP_VAL, MAX_CLIP_VAL)
    image = (image - NORM_MEAN) / NORM_STD

    if label is not None:
        image, label = pad_to_min_shape(image, label)
        image, label = adjust_shape_for_sliding_window(image, label)
        return image.astype(np.float32), label
    else:
        image = pad_to_min_shape(image)
        image = adjust_shape_for_sliding_window(image)
        return image.astype(np.float32)


def dice_score(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Compute Dice score for a specific class.

    DICE = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    Uses float64 precision per MLCommons reference accuracy_kits.py.
    """
    eps = 1e-6
    pred_mask = (prediction == class_id).astype(np.float64)
    target_mask = (target == class_id).astype(np.float64)
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
        super().__init__(data_path, count)
        self.data_path = Path(data_path)
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
        """Load preprocessed pickle files (MLCommons format).

        Supports two layouts:
        1. MLCommons format: {data_dir}/{case_name}.pkl containing [image, label]
        2. Directory format: {data_dir}/{case_name}/data.pkl + label.pkl
        """
        # First, try to use official MLPerf inference case list
        mlperf_cases_found = []
        for case_id in MLPERF_INFERENCE_CASES:
            # Check MLCommons format: case_XXXXX.pkl
            pkl_file = data_dir / f"{case_id}.pkl"
            if pkl_file.exists():
                mlperf_cases_found.append(case_id)
                continue
            # Check directory format: case_XXXXX/data.pkl or case_XXXXX/*.pkl
            case_dir = data_dir / case_id
            if case_dir.is_dir():
                pkl_files = list(case_dir.glob("*.pkl"))
                if pkl_files:
                    mlperf_cases_found.append(case_id)
                    continue

        if mlperf_cases_found:
            self.case_ids = mlperf_cases_found
        else:
            # Fall back to auto-discovery
            # Try direct pickle files first (MLCommons format)
            pkl_files = sorted(data_dir.glob("case_*.pkl"))
            if pkl_files:
                self.case_ids = [p.stem for p in pkl_files]
            else:
                # Try case directories
                case_dirs = sorted(data_dir.glob("case_*"))
                for case_dir in case_dirs:
                    if not case_dir.is_dir():
                        continue
                    pkl_file = case_dir / "data.pkl"
                    if not pkl_file.exists():
                        sub_pkls = list(case_dir.glob("*.pkl"))
                        if not sub_pkls:
                            continue
                    self.case_ids.append(case_dir.name)

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

        Returns dict with 'data' (preprocessed volume).
        """
        if index in self.samples:
            return self.samples[index]

        case_id = self.case_ids[index]
        return self._load_case(case_id, index)

    @staticmethod
    def _store_label(label: np.ndarray) -> np.ndarray:
        """Strip channel dim from label if present. Store as (D,H,W) for DICE."""
        if label.ndim == 4:
            return label[0]  # (1,D,H,W) -> (D,H,W)
        return label

    def _load_case(self, case_id: str, index: int) -> Dict[str, Any]:
        """Load and preprocess a single case.

        Supports:
        1. MLCommons pickle format: {case_id}.pkl containing [image(1,D,H,W), label(1,D,H,W)]
        2. Directory format: {case_id}/data.pkl + {case_id}/label.pkl
        3. Raw NIfTI format: {case_id}/imaging.nii.gz
        """
        preprocessed_dir = self.data_path / "preprocessed"
        raw_dir = self.data_path / "raw"

        # Try preprocessed pickle files
        for base_dir in [preprocessed_dir, self.data_path]:
            # MLCommons format: {case_id}.pkl with [image, label]
            pkl_path = base_dir / f"{case_id}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    image = data[0]
                    label = data[1]
                    sample = {"data": image, "case_id": case_id}
                    self.samples[index] = sample
                    if label is not None:
                        self.labels[index] = self._store_label(label)
                    return sample

            # Directory format: {case_id}/data.pkl
            pkl_path = base_dir / case_id / "data.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    image = data[0]
                    label = data[1] if len(data) > 1 else None
                elif isinstance(data, dict) and "data" in data:
                    image = data["data"]
                    label = None
                else:
                    image = data
                    label = None

                sample = {"data": image, "case_id": case_id}
                self.samples[index] = sample

                if label is not None:
                    self.labels[index] = self._store_label(label)
                else:
                    # Try separate label file
                    label_path = base_dir / case_id / "label.pkl"
                    if label_path.exists():
                        with open(label_path, "rb") as f:
                            raw_label = pickle.load(f)
                        self.labels[index] = self._store_label(raw_label)

                return sample

            # Try any .pkl file in the case directory
            case_dir = base_dir / case_id
            if case_dir.is_dir():
                pkl_files = sorted(case_dir.glob("*.pkl"))
                for pf in pkl_files:
                    with open(pf, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, (list, tuple)) and len(data) >= 2:
                        image = data[0]
                        label = data[1]
                    elif isinstance(data, dict) and "data" in data:
                        image = data["data"]
                        label = None
                    else:
                        image = data
                        label = None

                    sample = {"data": image, "case_id": case_id}
                    self.samples[index] = sample
                    if label is not None:
                        self.labels[index] = self._store_label(label)
                    return sample

        # Try NIfTI
        for base_dir in [raw_dir, self.data_path]:
            nii_path = base_dir / case_id / "imaging.nii.gz"
            if nii_path.exists():
                return self._load_nifti_case(nii_path, case_id, index, base_dir)

        raise FileNotFoundError(f"Case {case_id} not found in {self.data_path}")

    def _load_nifti_case(self, nii_path: Path, case_id: str, index: int, base_dir: Path) -> Dict[str, Any]:
        """Load and preprocess a NIfTI case per MLCommons reference pipeline.

        Pipeline: resample → channel dim → normalize → pad_to_min_shape → adjust_shape_for_sliding_window.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required for loading NIfTI files: pip install nibabel")

        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.float32)

        # Resample to target spacing
        original_spacing = img.header.get_zooms()[:3]
        data = self._resample_volume(data, original_spacing, TARGET_SPACING, order=1)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        data = data[np.newaxis, ...]

        # Normalize intensity
        data = np.clip(data, MIN_CLIP_VAL, MAX_CLIP_VAL)
        data = (data - NORM_MEAN) / NORM_STD

        # Load segmentation label if available
        label = None
        label_path = base_dir / case_id / "segmentation.nii.gz"
        if label_path.exists():
            label_img = nib.load(str(label_path))
            label = label_img.get_fdata().astype(np.uint8)
            label = self._resample_volume(label, original_spacing, TARGET_SPACING, order=0)
            label = label[np.newaxis, ...]  # (1, D, H, W)

        # Pad to min shape (symmetric edge)
        if label is not None:
            data, label = pad_to_min_shape(data, label)
        else:
            data = pad_to_min_shape(data)

        # Adjust shape for sliding window (crop + constant pad)
        if label is not None:
            data, label = adjust_shape_for_sliding_window(data, label)
        else:
            data = adjust_shape_for_sliding_window(data)

        sample = {
            "data": data,
            "case_id": case_id,
        }
        self.samples[index] = sample

        if label is not None:
            self.labels[index] = label[0]  # Store as (D, H, W) for DICE computation

        return sample

    @staticmethod
    def _resample_volume(
        data: np.ndarray,
        original_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...],
        order: int = 1,
    ) -> np.ndarray:
        """Resample a 3D volume to target voxel spacing.

        Per MLCommons reference: mode='constant', cval=data.min().
        """
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError("scipy is required for resampling: pip install scipy")

        zoom_factors = tuple(o / t for o, t in zip(original_spacing, target_spacing))
        if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
            return data
        return zoom(
            data, zoom_factors, order=order,
            mode="constant", cval=data.min(),
        ).astype(data.dtype)

    def get_samples(self, indices: List[int]) -> Tuple[List[Dict[str, Any]], List[Optional[np.ndarray]]]:
        """Get multiple preprocessed samples by indices.

        Returns (samples, labels) tuple. 3D volumes have variable spatial shapes,
        so samples are returned as a list of dicts rather than a stacked ndarray.
        """
        samples = []
        labels = []
        for idx in indices:
            samples.append(self.get_sample(idx))
            labels.append(self.get_label(idx))
        return samples, labels

    def get_label(self, index: int) -> Optional[np.ndarray]:
        """Get ground truth segmentation label for a sample."""
        return self.labels.get(index, None)

    def postprocess(self, result: np.ndarray, sample_indices: List[int]) -> List[np.ndarray]:
        """Postprocess model output: argmax over class dimension.

        Uses uint8 per MLCommons reference inference_utils.py apply_argmax().
        """
        # result shape: (num_classes, D, H, W) -> (D, H, W) via argmax
        if result.ndim == 4:
            return [np.argmax(result, axis=0).astype(np.uint8)]
        elif result.ndim == 5:
            # Batched: (B, C, D, H, W)
            return [np.argmax(result[i], axis=0).astype(np.uint8) for i in range(result.shape[0])]
        return [result.astype(np.uint8)]

    def compute_accuracy(self, predictions: List[np.ndarray], labels: List[np.ndarray]) -> Dict[str, Any]:
        """Compute mean Dice score (kidney + tumor)."""
        kidney_dices = []
        tumor_dices = []

        for pred, label in zip(predictions, labels):
            if label is None:
                continue
            # Crop prediction to label shape (undo inference padding)
            if pred.shape != label.shape:
                pred = pred[:label.shape[0], :label.shape[1], :label.shape[2]]
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
        performance_sample_count: int = 43,
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

    def get_features(self, sample_id: int) -> Dict[str, Any]:
        """Get preprocessed features for a sample.

        Returns dict with 'input' as (1, 1, D, H, W) tensor.
        Preprocessed pickle data (1, D, H, W) is already padded and adjusted
        for sliding window, matching MLCommons reference format.
        """
        sample = self._loaded_samples.get(sample_id)
        if sample is None:
            sample = self.dataset.get_sample(sample_id)
            self._loaded_samples[sample_id] = sample

        data = sample["data"]

        if data.ndim == 3:
            # (D, H, W) — old pickle format: already normalized, but no padding/adjustment
            # Only pad and adjust (do NOT re-normalize)
            data = data[np.newaxis, ...]  # (1, D, H, W)
            data = pad_to_min_shape(data)
            data = adjust_shape_for_sliding_window(data)
            data = data[np.newaxis, ...]  # (1, 1, D', H', W')
        elif data.ndim == 4:
            # (1, D, H, W) — reference format: already normalized, padded, adjusted
            data = data[np.newaxis, ...]  # (1, 1, D, H, W)
        else:
            # (1, 1, D, H, W) — already batched
            pass

        return {"input": data.astype(np.float32)}

    def get_label(self, sample_id: int) -> Optional[np.ndarray]:
        """Get ground truth label."""
        return self.dataset.get_label(sample_id)

    @property
    def total_sample_count(self) -> int:
        return self.dataset.total_count

    @property
    def performance_sample_count(self) -> int:
        # MLPerf 3D UNET: use all samples (43) for performance
        return min(self._performance_sample_count, self.dataset.total_count)
