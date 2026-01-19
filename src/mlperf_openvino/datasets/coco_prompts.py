"""
COCO 2014 Captions dataset for Stable Diffusion XL benchmark.

This module provides dataset handling for the COCO 2014 captions dataset
used in MLPerf Inference for SDXL (text-to-image) model evaluation.

MLPerf uses a subset of 5000 images and captions from COCO 2014 validation set,
with exactly one caption per image.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# SDXL constants
MAX_SEQUENCE_LENGTH = 77  # Maximum token length for CLIP text encoder
IMAGE_SIZE = 1024  # SDXL generates 1024x1024 images
LATENT_SIZE = 128  # Latent space size (1024 / 8)
GUIDANCE_SCALE = 8.0  # Default classifier-free guidance scale
NUM_INFERENCE_STEPS = 20  # Default number of denoising steps


class COCOPromptsDataset(BaseDataset):
    """
    COCO 2014 Captions dataset for SDXL benchmark.

    This dataset provides text prompts (captions) from COCO 2014 for
    text-to-image generation benchmarking. For accuracy evaluation,
    it also provides reference images for FID calculation.

    Expected directory structure:
        data_path/
        ├── captions/
        │   └── captions_val2014.json (or captions.json)
        ├── images/
        │   └── val2014/
        │       ├── COCO_val2014_000000000042.jpg
        │       └── ...
        └── latents/ (optional, for pre-computed noise latents)
            ├── latents.pt
            └── ...

    Or MLCommons format:
        data_path/
        ├── coco-1024.tsv (prompt file with format: id<tab>prompt)
        └── coco-1024/ (reference images resized to 1024x1024)
            ├── 000000000042.png
            └── ...
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        use_latents: bool = False,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ):
        """
        Initialize COCO prompts dataset.

        Args:
            data_path: Path to dataset directory
            count: Number of samples to use (None = all, default 5000 for MLPerf)
            use_latents: Whether to use pre-computed latents for reproducibility
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion denoising steps
        """
        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.use_latents = use_latents
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        self._samples: List[Dict[str, Any]] = []
        self._latents_cache: Dict[int, np.ndarray] = {}
        self._is_loaded = False

        # Tokenizer (lazy loaded)
        self._tokenizer = None

    def _load_tokenizer(self):
        """Load CLIP tokenizer for text encoding."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            logger.info("Loaded CLIP tokenizer")
        except ImportError:
            logger.warning(
                "transformers not installed. Install with: pip install transformers"
            )
            self._tokenizer = None

    def load(self) -> None:
        """Load dataset metadata."""
        if self._is_loaded:
            return

        logger.info(f"Loading COCO prompts dataset from {self.data_path}")

        # Try different formats
        loaded = False

        # Format 1: MLCommons TSV format (coco-1024.tsv)
        tsv_file = self.data_path / "coco-1024.tsv"
        if not tsv_file.exists():
            tsv_file = self.data_path / "captions.tsv"
        if tsv_file.exists():
            self._load_from_tsv(tsv_file)
            loaded = True

        # Format 2: COCO JSON annotations format
        if not loaded:
            json_candidates = [
                self.data_path / "captions" / "captions_val2014.json",
                self.data_path / "annotations" / "captions_val2014.json",
                self.data_path / "captions.json",
                self.data_path / "captions_val2014.json",
            ]
            for json_file in json_candidates:
                if json_file.exists():
                    self._load_from_coco_json(json_file)
                    loaded = True
                    break

        # Format 3: Simple text file (one caption per line)
        if not loaded:
            txt_candidates = [
                self.data_path / "prompts.txt",
                self.data_path / "captions.txt",
            ]
            for txt_file in txt_candidates:
                if txt_file.exists():
                    self._load_from_txt(txt_file)
                    loaded = True
                    break

        if not self._samples:
            logger.warning(f"No samples found in {self.data_path}")
            logger.info("For COCO dataset, expected structure:")
            logger.info("  data_path/")
            logger.info("  ├── coco-1024.tsv (MLCommons format)")
            logger.info("  └── coco-1024/ (reference images)")
            logger.info("")
            logger.info("Or use mlperf-ov download-dataset --dataset coco2014")

        # Limit count if specified (MLPerf uses 5000 samples)
        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        # Load pre-computed latents if available
        if self.use_latents:
            self._load_latents()

        logger.info(f"Loaded {len(self._samples)} prompts")
        self._is_loaded = True

    def _load_from_tsv(self, tsv_file: Path) -> None:
        """Load samples from TSV format (MLCommons format)."""
        logger.info(f"Loading from TSV: {tsv_file}")

        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    image_id = parts[0]
                    caption = parts[1]
                else:
                    # Single column (caption only)
                    image_id = str(line_num)
                    caption = parts[0]

                # Find reference image if available
                image_path = self._find_reference_image(image_id)

                self._samples.append({
                    "id": image_id,
                    "caption": caption,
                    "image_path": image_path,
                })

    def _load_from_coco_json(self, json_file: Path) -> None:
        """Load samples from COCO JSON annotations format."""
        logger.info(f"Loading from COCO JSON: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build image_id to filename mapping
        id_to_filename = {}
        if 'images' in data:
            for img in data['images']:
                id_to_filename[img['id']] = img.get('file_name', '')

        # For MLPerf, we need one caption per image
        # Take the first caption for each image
        seen_images = set()
        annotations = data.get('annotations', [])

        for ann in annotations:
            image_id = ann.get('image_id')
            if image_id in seen_images:
                continue

            seen_images.add(image_id)
            caption = ann.get('caption', '')

            # Find reference image
            filename = id_to_filename.get(image_id, f"{image_id:012d}.jpg")
            image_path = self._find_reference_image(str(image_id), filename)

            self._samples.append({
                "id": str(image_id),
                "caption": caption,
                "image_path": image_path,
            })

    def _load_from_txt(self, txt_file: Path) -> None:
        """Load samples from simple text file (one caption per line)."""
        logger.info(f"Loading from TXT: {txt_file}")

        with open(txt_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                caption = line.strip()
                if not caption:
                    continue

                self._samples.append({
                    "id": str(idx),
                    "caption": caption,
                    "image_path": None,  # No reference images
                })

    def _find_reference_image(
        self,
        image_id: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Find reference image for accuracy computation."""
        # Search in various locations
        search_dirs = [
            self.data_path / "coco-1024",  # MLCommons format
            self.data_path / "images" / "val2014",  # COCO format
            self.data_path / "val2014",
            self.data_path / "images",
        ]

        # Possible filenames
        filenames = []
        if filename:
            filenames.append(filename)
        filenames.extend([
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"COCO_val2014_{int(image_id):012d}.jpg" if image_id.isdigit() else None,
            f"{int(image_id):012d}.png" if image_id.isdigit() else None,
        ])
        filenames = [f for f in filenames if f]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for fname in filenames:
                candidate = search_dir / fname
                if candidate.exists():
                    return str(candidate)

        return None

    def _load_latents(self) -> None:
        """Load pre-computed latents for reproducibility."""
        latents_file = self.data_path / "latents" / "latents.pt"
        if not latents_file.exists():
            latents_file = self.data_path / "latents.pt"

        if latents_file.exists():
            try:
                import torch
                latents = torch.load(latents_file)
                for idx, lat in enumerate(latents):
                    if idx < len(self._samples):
                        self._latents_cache[idx] = lat.numpy()
                logger.info(f"Loaded {len(self._latents_cache)} pre-computed latents")
            except Exception as e:
                logger.warning(f"Failed to load latents: {e}")

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def total_count(self) -> int:
        return len(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Tuple[Dict[str, Any], str]:
        """
        Get a sample (prompt and metadata).

        Args:
            index: Sample index

        Returns:
            Tuple of (input_dict, caption)
            input_dict contains:
            - 'prompt': text caption
            - 'latents': pre-computed latents (if available)
            - 'image_path': reference image path (if available)
        """
        sample = self._samples[index]
        caption = sample['caption']

        input_dict = {
            'prompt': caption,
            'guidance_scale': self.guidance_scale,
            'num_inference_steps': self.num_inference_steps,
        }

        # Add pre-computed latents if available
        if index in self._latents_cache:
            input_dict['latents'] = self._latents_cache[index]

        # Add reference image path for accuracy
        if sample.get('image_path'):
            input_dict['reference_image_path'] = sample['image_path']

        return input_dict, caption

    def get_prompt(self, index: int) -> str:
        """Get caption/prompt for sample."""
        return self._samples[index]['caption']

    def get_reference_image_path(self, index: int) -> Optional[str]:
        """Get reference image path for accuracy calculation."""
        return self._samples[index].get('image_path')

    def tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """
        Tokenize text prompt for CLIP text encoder.

        Args:
            text: Text prompt

        Returns:
            Dictionary with input_ids and attention_mask
        """
        self._load_tokenizer()

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available")

        tokens = self._tokenizer(
            text,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="np",
        )

        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        }

    def postprocess(
        self,
        results: np.ndarray,
        indices: List[int]
    ) -> List[np.ndarray]:
        """
        Postprocess model outputs (generated images).

        Args:
            results: Generated images as numpy arrays
            indices: Sample indices

        Returns:
            List of processed images
        """
        # Convert from latent space to pixel space if needed
        # Results should already be in [0, 255] uint8 format
        if isinstance(results, list):
            return results
        return [results[i] for i in range(len(indices))]

    def compute_accuracy(
        self,
        generated_images: List[np.ndarray],
        indices: List[int]
    ) -> Dict[str, float]:
        """
        Compute CLIP score and FID for generated images.

        MLPerf v5.1 accuracy targets for SDXL (closed division):
        - CLIP_SCORE: >= 31.68632 and <= 31.81332
        - FID_SCORE: >= 23.01086 and <= 23.95007

        Args:
            generated_images: List of generated images
            indices: Sample indices

        Returns:
            Dictionary with CLIP score and FID
        """
        metrics = {
            'clip_score': 0.0,
            'fid_score': 0.0,
            'num_samples': len(generated_images),
        }

        if not generated_images:
            return metrics

        # Compute CLIP score
        clip_score = self._compute_clip_score(generated_images, indices)
        metrics['clip_score'] = clip_score

        # Compute FID if reference images are available
        reference_images = []
        for idx in indices:
            ref_path = self.get_reference_image_path(idx)
            if ref_path:
                reference_images.append(ref_path)

        if reference_images:
            fid_score = self._compute_fid_score(generated_images, reference_images)
            metrics['fid_score'] = fid_score

        # Check MLPerf compliance
        metrics['clip_score_valid'] = 31.68632 <= clip_score <= 31.81332
        metrics['fid_score_valid'] = 23.01086 <= metrics['fid_score'] <= 23.95007

        return metrics

    def _compute_clip_score(
        self,
        images: List[np.ndarray],
        indices: List[int]
    ) -> float:
        """
        Compute CLIP score between generated images and prompts.

        Args:
            images: Generated images
            indices: Sample indices for getting prompts

        Returns:
            Average CLIP score
        """
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            logger.warning("transformers not installed, cannot compute CLIP score")
            return 0.0

        try:
            # Load CLIP model
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model.eval()

            scores = []
            for img, idx in zip(images, indices):
                prompt = self.get_prompt(idx)

                # Prepare inputs
                inputs = processor(
                    text=[prompt],
                    images=[img],
                    return_tensors="pt",
                    padding=True
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    # CLIP score is the cosine similarity * 100
                    logits_per_image = outputs.logits_per_image
                    score = logits_per_image.item()
                    scores.append(score)

            return float(np.mean(scores)) if scores else 0.0

        except Exception as e:
            logger.warning(f"Error computing CLIP score: {e}")
            return 0.0

    def _compute_fid_score(
        self,
        generated_images: List[np.ndarray],
        reference_paths: List[str]
    ) -> float:
        """
        Compute FID score between generated and reference images.

        Args:
            generated_images: Generated images
            reference_paths: Paths to reference images

        Returns:
            FID score
        """
        try:
            from scipy import linalg
            import torch
            from torchvision import transforms
            from PIL import Image
        except ImportError:
            logger.warning("Required packages not installed for FID computation")
            return 0.0

        try:
            # Load InceptionV3 for feature extraction
            from torchvision.models import inception_v3, Inception_V3_Weights
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            model.fc = torch.nn.Identity()  # Remove final FC layer
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            def get_features(images_or_paths, is_path=False):
                features = []
                for item in images_or_paths:
                    if is_path:
                        img = Image.open(item).convert('RGB')
                    else:
                        if isinstance(item, np.ndarray):
                            img = Image.fromarray(item.astype(np.uint8))
                        else:
                            img = item

                    tensor = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        feat = model(tensor).numpy().flatten()
                    features.append(feat)
                return np.array(features)

            # Get features
            gen_features = get_features(generated_images, is_path=False)
            ref_features = get_features(reference_paths, is_path=True)

            # Compute statistics
            mu_gen = np.mean(gen_features, axis=0)
            mu_ref = np.mean(ref_features, axis=0)
            sigma_gen = np.cov(gen_features, rowvar=False)
            sigma_ref = np.cov(ref_features, rowvar=False)

            # Compute FID
            diff = mu_gen - mu_ref
            covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
            return float(fid)

        except Exception as e:
            logger.warning(f"Error computing FID score: {e}")
            return 0.0


class COCOPromptsQSL(QuerySampleLibrary):
    """
    Query Sample Library for COCO prompts dataset.

    Implements the MLPerf LoadGen QSL interface for SDXL benchmark.
    """

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        performance_sample_count: int = 5000,  # MLPerf default
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ):
        """
        Initialize COCO prompts QSL.

        Args:
            data_path: Path to dataset directory
            count: Number of samples to use
            performance_sample_count: Number of samples for performance run
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
        """
        super().__init__()

        self.dataset = COCOPromptsDataset(
            data_path=data_path,
            count=count,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
        """
        Load samples into memory.

        Args:
            sample_indices: Indices of samples to load
        """
        for idx in sample_indices:
            if idx not in self._loaded_samples:
                input_dict, _ = self.dataset.get_sample(idx)
                self._loaded_samples[idx] = input_dict

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """
        Unload samples from memory.

        Args:
            sample_indices: Indices of samples to unload
        """
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, Any]:
        """
        Get input features for a sample.

        Args:
            sample_index: Sample index

        Returns:
            Dictionary with prompt and generation parameters
        """
        if sample_index in self._loaded_samples:
            return self._loaded_samples[sample_index]
        else:
            input_dict, _ = self.dataset.get_sample(sample_index)
            return input_dict

    def get_prompt(self, sample_index: int) -> str:
        """
        Get text prompt for a sample.

        Args:
            sample_index: Sample index

        Returns:
            Text prompt/caption
        """
        return self.dataset.get_prompt(sample_index)

    def get_label(self, sample_index: int) -> str:
        """
        Get ground truth label (prompt) for a sample.

        Args:
            sample_index: Sample index

        Returns:
            Text prompt (used for CLIP score computation)
        """
        return self.dataset.get_prompt(sample_index)
