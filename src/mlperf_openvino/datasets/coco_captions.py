"""
COCO 2014 Captions dataset for SDXL text-to-image benchmark.

This module provides dataset implementation for MLPerf SDXL benchmark
using COCO 2014 captions as prompts for image generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# MLPerf SDXL constants
NUM_INFERENCE_STEPS = 20  # Default denoising steps
GUIDANCE_SCALE = 8.0  # CFG scale
IMAGE_SIZE = 1024  # Output image resolution


class COCOCaptionsDataset(BaseDataset):
    """
    COCO 2014 Captions dataset for SDXL benchmark.

    Expected directory structure:
        data_path/
            captions/
                captions.tsv    # Tab-separated: id, caption
            latents/
                latents.pt      # Pre-generated latents for determinism
            images/             # Reference images (for accuracy)
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        latent_dtype: str = "float16",
        **kwargs
    ):
        """
        Initialize COCO Captions dataset.

        Args:
            data_path: Path to dataset directory
            count: Number of samples to use (None = all, default 5000)
            latent_dtype: Data type for latents (float16 or float32)
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required. Install with: pip install pandas")

        super().__init__(data_path=data_path, count=count, **kwargs)

        self.data_path = Path(data_path)
        self.latent_dtype = latent_dtype

        self._captions: List[str] = []
        self._ids: List[str] = []
        self._latents: Optional[np.ndarray] = None
        self._is_loaded = False

    def load(self) -> None:
        """Load dataset metadata and latents."""
        if self._is_loaded:
            return

        logger.info(f"Loading COCO Captions dataset from {self.data_path}")

        # Load captions from TSV
        captions_file = self.data_path / "captions" / "captions.tsv"
        if not captions_file.exists():
            # Try alternative location
            captions_file = self.data_path / "captions.tsv"

        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")

        logger.info(f"Loading captions from {captions_file}")
        df = pd.read_csv(captions_file, sep="\t")

        # Extract captions and IDs
        if "caption" in df.columns:
            self._captions = df["caption"].tolist()
        elif "prompt" in df.columns:
            self._captions = df["prompt"].tolist()
        else:
            # Assume second column is caption
            self._captions = df.iloc[:, 1].tolist()

        if "id" in df.columns:
            self._ids = df["id"].astype(str).tolist()
        elif "image_id" in df.columns:
            self._ids = df["image_id"].astype(str).tolist()
        else:
            # Generate IDs
            self._ids = [str(i) for i in range(len(self._captions))]

        # Apply count limit
        if self.count and self.count > 0:
            self._captions = self._captions[:self.count]
            self._ids = self._ids[:self.count]

        logger.info(f"Loaded {len(self._captions)} captions")

        # Load pre-generated latents if available
        self._load_latents()

        self._is_loaded = True
        self._loaded = True

    def _load_latents(self) -> None:
        """Load pre-generated latents for deterministic generation."""
        latents_dir = self.data_path / "latents"

        # Try PyTorch format first
        pt_file = latents_dir / "latents.pt"
        if pt_file.exists() and TORCH_AVAILABLE:
            logger.info(f"Loading latents from {pt_file}")
            import torch
            latents = torch.load(pt_file, map_location="cpu")
            if self.latent_dtype == "float16":
                latents = latents.half()
            self._latents = latents.numpy()
            logger.info(f"Loaded latents shape: {self._latents.shape}")
            return

        # Try NumPy format
        npy_file = latents_dir / "latents.npy"
        if npy_file.exists():
            logger.info(f"Loading latents from {npy_file}")
            self._latents = np.load(npy_file)
            if self.latent_dtype == "float16":
                self._latents = self._latents.astype(np.float16)
            logger.info(f"Loaded latents shape: {self._latents.shape}")
            return

        logger.warning("No pre-generated latents found. Will use random latents.")
        self._latents = None

    @property
    def total_count(self) -> int:
        """Total number of samples."""
        return len(self._captions)

    def get_sample(self, index: int) -> Tuple[Dict[str, Any], str]:
        """
        Get a sample by index.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample_dict, caption_id)
        """
        if not self._is_loaded:
            self.load()

        caption = self._captions[index]
        sample_id = self._ids[index]

        # Get latent for this sample
        latent = None
        if self._latents is not None and index < len(self._latents):
            latent = self._latents[index]

        return {
            "prompt": caption,
            "latent": latent,
            "index": index,
        }, sample_id

    def get_caption(self, index: int) -> str:
        """Get caption for a sample."""
        if not self._is_loaded:
            self.load()
        return self._captions[index]

    def get_latent(self, index: int) -> Optional[np.ndarray]:
        """Get pre-generated latent for a sample."""
        if self._latents is None:
            return None
        if index >= len(self._latents):
            return None
        return self._latents[index]

    def compute_accuracy(
        self,
        generated_images: List[np.ndarray],
        indices: List[int]
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics (FID and CLIP score).

        This is a placeholder - actual computation requires:
        - Reference images for FID
        - CLIP model for CLIP score

        Args:
            generated_images: List of generated images
            indices: Sample indices

        Returns:
            Dictionary with FID and CLIP scores
        """
        # Placeholder - actual implementation needs FID and CLIP computation
        logger.warning("Accuracy computation not fully implemented yet")
        return {
            "fid": 0.0,
            "clip_score": 0.0,
            "num_samples": len(generated_images),
        }


class COCOCaptionsQSL(QuerySampleLibrary):
    """
    Query Sample Library for COCO Captions dataset.

    Implements the MLPerf LoadGen QSL interface for SDXL benchmark.
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        performance_sample_count: int = 5000,  # MLPerf default
        latent_dtype: str = "float16",
    ):
        """
        Initialize COCO Captions QSL.

        Args:
            data_path: Path to dataset directory
            count: Number of samples to use
            performance_sample_count: Number of samples for performance run
            latent_dtype: Data type for latents
        """
        super().__init__()

        self.dataset = COCOCaptionsDataset(
            data_path=data_path,
            count=count,
            latent_dtype=latent_dtype,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, Dict[str, Any]] = {}

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
        """Load samples into memory."""
        if not self.dataset._is_loaded:
            self.dataset.load()

        logger.info(f"Loading {len(sample_indices)} query samples...")

        for idx in sample_indices:
            if idx not in self._loaded_samples:
                sample, _ = self.dataset.get_sample(idx)
                self._loaded_samples[idx] = sample

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """Unload samples from memory."""
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, index: int) -> Dict[str, Any]:
        """Get features for a sample."""
        if index in self._loaded_samples:
            return self._loaded_samples[index]

        sample, _ = self.dataset.get_sample(index)
        return sample

    def get_label(self, index: int) -> str:
        """Get caption/prompt for a sample."""
        return self.dataset.get_caption(index)
