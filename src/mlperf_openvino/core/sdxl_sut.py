"""
Stable Diffusion XL System Under Test implementation.

This module provides SUT implementation for SDXL text-to-image model,
using OpenVINO for inference or optimum-intel OVStableDiffusionXLPipeline.
"""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

# Optimum-Intel for SDXL pipeline
try:
    from optimum.intel import OVStableDiffusionXLPipeline
    OPTIMUM_SDXL_AVAILABLE = True
except ImportError:
    OPTIMUM_SDXL_AVAILABLE = False
    OVStableDiffusionXLPipeline = None

# Diffusers for alternative pipeline
try:
    from diffusers import StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionXLPipeline = None

from .config import BenchmarkConfig, Scenario
from ..datasets.coco_prompts import COCOPromptsQSL

logger = logging.getLogger(__name__)

# SDXL default parameters (MLCommons reference implementation)
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGE_SIZE = 1024
# MLCommons official negative prompt
DEFAULT_NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"


class SDXLOptimumSUT:
    """
    System Under Test for Stable Diffusion XL using Optimum-Intel.

    Uses OVStableDiffusionXLPipeline for optimized OpenVINO inference
    with proper handling of UNet, VAE, and text encoders.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: COCOPromptsQSL,
        scenario: Scenario = Scenario.OFFLINE,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        """
        Initialize SDXL SUT using Optimum-Intel.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO SDXL model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            image_size: Output image size (default 1024x1024)
            negative_prompt: Negative prompt for guidance
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_SDXL_AVAILABLE:
            raise ImportError(
                "Optimum-Intel with SDXL support is required. "
                "Install with: pip install optimum[openvino] diffusers"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size
        self.negative_prompt = negative_prompt

        # Results storage - store generated images
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 1.0  # seconds (SDXL is slower)

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Random generator for reproducibility
        self._generator = None

        # Load pipeline
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load SDXL pipeline using Optimum-Intel."""
        logger.info(f"Loading SDXL model from {self.model_path}")

        try:
            # Load OpenVINO optimized pipeline
            self.pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                str(self.model_path),
                compile=True,
            )
            logger.info("SDXL pipeline loaded successfully (OpenVINO)")

        except Exception as e:
            logger.warning(f"Failed to load OpenVINO pipeline: {e}")

            if DIFFUSERS_AVAILABLE:
                logger.info("Falling back to diffusers pipeline")
                import torch
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float32,
                )
                # Move to CPU
                self.pipeline = self.pipeline.to("cpu")
            else:
                raise RuntimeError(
                    f"Cannot load SDXL model from {self.model_path}: {e}"
                )

    def _get_generator(self, seed: int = 42):
        """Get random generator for reproducibility."""
        try:
            import torch
            return torch.Generator().manual_seed(seed)
        except ImportError:
            return None

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="images",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} images)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {self._sample_count} images, "
                    f"{throughput:.2f} images/sec"
                )
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completed: {self._sample_count} images in {elapsed:.1f}s "
                f"({throughput:.2f} images/sec)"
            )

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """
        Process a single prompt and generate an image.

        Args:
            sample_idx: Sample index

        Returns:
            Generated image as numpy array (H, W, C) in uint8
        """
        # Get prompt from QSL
        features = self.qsl.get_features(sample_idx)
        prompt = features['prompt']

        # Get generation parameters (may be overridden per sample)
        guidance_scale = features.get('guidance_scale', self.guidance_scale)
        num_steps = features.get('num_inference_steps', self.num_inference_steps)

        # Use pre-computed latents if available (for reproducibility)
        latents = features.get('latents', None)

        # Convert numpy latents to torch tensor if needed
        # The pipeline expects torch tensors, not numpy arrays
        if latents is not None:
            try:
                import torch
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents).float()
            except ImportError:
                logger.warning("PyTorch not available, cannot convert latents")
                latents = None

        # Generate image
        generator = self._get_generator(seed=sample_idx)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            height=self.image_size,
            width=self.image_size,
            generator=generator,
            latents=latents,
        )

        # Extract image from result
        image = result.images[0]

        # Convert PIL Image to numpy array
        if hasattr(image, 'numpy'):
            image_array = np.array(image)
        else:
            # PIL Image
            image_array = np.array(image)

        return image_array

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="SDXL Offline generation")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Generate image
            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image

            # Create response (store image size as dummy response)
            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc="SDXL Server generation")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Generate image
            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image

            # Create response
            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all generated images."""
        return self._predictions.copy()

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute CLIP score and FID for generated images."""
        predictions = self.get_predictions()

        if not predictions:
            return {
                'clip_score': 0.0,
                'fid_score': 0.0,
                'num_samples': 0
            }

        # Prepare images and indices
        indices = sorted(predictions.keys())
        images = [predictions[idx] for idx in indices]

        # Use dataset's compute_accuracy
        return self.qsl.dataset.compute_accuracy(images, indices)

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class SDXLManualSUT:
    """
    System Under Test for SDXL with manual component loading.

    This SUT loads individual SDXL components (UNet, VAE, text encoders)
    as separate OpenVINO models for more control over the inference pipeline.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: COCOPromptsQSL,
        scenario: Scenario = Scenario.OFFLINE,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        """
        Initialize SDXL SUT with manual component loading.

        Args:
            config: Benchmark configuration
            model_path: Path to model directory with SDXL components
            qsl: Query Sample Library
            scenario: MLPerf scenario
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            image_size: Output image size
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size

        # Results storage
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 1.0

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Model components
        self.unet = None
        self.vae_decoder = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.scheduler = None

        # Load components
        self._load_components()

    def _load_components(self) -> None:
        """Load SDXL model components."""
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("OpenVINO is required for SDXL inference")

        core = ov.Core()
        logger.info(f"Loading SDXL components from {self.model_path}")

        # Load UNet
        unet_path = self.model_path / "unet" / "openvino_model.xml"
        if not unet_path.exists():
            unet_path = self.model_path / "unet.xml"
        if unet_path.exists():
            logger.info(f"Loading UNet from {unet_path}")
            self.unet = core.compile_model(str(unet_path), "CPU")

        # Load VAE decoder
        vae_path = self.model_path / "vae_decoder" / "openvino_model.xml"
        if not vae_path.exists():
            vae_path = self.model_path / "vae_decoder.xml"
        if vae_path.exists():
            logger.info(f"Loading VAE decoder from {vae_path}")
            self.vae_decoder = core.compile_model(str(vae_path), "CPU")

        # Load text encoder
        text_enc_path = self.model_path / "text_encoder" / "openvino_model.xml"
        if not text_enc_path.exists():
            text_enc_path = self.model_path / "text_encoder.xml"
        if text_enc_path.exists():
            logger.info(f"Loading text encoder from {text_enc_path}")
            self.text_encoder = core.compile_model(str(text_enc_path), "CPU")

        # Load text encoder 2 (SDXL uses two text encoders)
        text_enc2_path = self.model_path / "text_encoder_2" / "openvino_model.xml"
        if not text_enc2_path.exists():
            text_enc2_path = self.model_path / "text_encoder_2.xml"
        if text_enc2_path.exists():
            logger.info(f"Loading text encoder 2 from {text_enc2_path}")
            self.text_encoder_2 = core.compile_model(str(text_enc2_path), "CPU")

        # Load tokenizers
        try:
            from transformers import CLIPTokenizer
            tokenizer_path = self.model_path / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path))
            else:
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )

            tokenizer2_path = self.model_path / "tokenizer_2"
            if tokenizer2_path.exists():
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(str(tokenizer2_path))
            else:
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                )
        except ImportError:
            logger.warning("transformers not available, tokenizers not loaded")

        # Load scheduler
        try:
            from diffusers import EulerDiscreteScheduler
            scheduler_path = self.model_path / "scheduler"
            if scheduler_path.exists():
                self.scheduler = EulerDiscreteScheduler.from_pretrained(
                    str(scheduler_path)
                )
            else:
                self.scheduler = EulerDiscreteScheduler.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="scheduler"
                )
        except ImportError:
            logger.warning("diffusers not available, using simple scheduler")
            self.scheduler = None

        if self.unet is None or self.vae_decoder is None:
            raise RuntimeError(
                f"Required SDXL components not found in {self.model_path}. "
                f"Expected: unet.xml, vae_decoder.xml"
            )

        logger.info("SDXL components loaded successfully")

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode text prompt using CLIP text encoders."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np"
        )

        # Encode with first text encoder
        text_input_ids = tokens['input_ids']

        if self.text_encoder is not None:
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        else:
            # Fallback: zero embeddings
            prompt_embeds = np.zeros((1, 77, 768), dtype=np.float32)

        # Encode with second text encoder (SDXL specific)
        if self.text_encoder_2 is not None and self.tokenizer_2 is not None:
            tokens_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np"
            )
            pooled_output = self.text_encoder_2(tokens_2['input_ids'])
            # Get pooled and hidden states
            if isinstance(pooled_output, tuple):
                prompt_embeds_2 = pooled_output[0]
                pooled_embeds = pooled_output[1] if len(pooled_output) > 1 else None
            else:
                prompt_embeds_2 = pooled_output
                pooled_embeds = None

            # Concatenate embeddings
            prompt_embeds = np.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)

        return prompt_embeds

    def _generate_latents(self, batch_size: int = 1) -> np.ndarray:
        """Generate initial random latents."""
        # SDXL uses 128x128 latent space for 1024x1024 images
        latent_size = self.image_size // 8
        latents = np.random.randn(
            batch_size, 4, latent_size, latent_size
        ).astype(np.float32)
        return latents

    def _denoise_step(
        self,
        latents: np.ndarray,
        prompt_embeds: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Perform one denoising step with UNet."""
        # Prepare UNet inputs
        latent_input = np.concatenate([latents] * 2)  # For CFG
        timestep_array = np.array([timestep], dtype=np.float32)

        # Run UNet
        noise_pred = self.unet({
            'sample': latent_input,
            'timestep': timestep_array,
            'encoder_hidden_states': prompt_embeds,
        })[0]

        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred

    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents to image using VAE decoder."""
        # Scale latents
        latents = latents / 0.18215

        # Decode
        image = self.vae_decoder(latents)[0]

        # Post-process
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image * 255).astype(np.uint8)

        # NCHW to NHWC
        if image.ndim == 4:
            image = image.transpose(0, 2, 3, 1)[0]
        elif image.ndim == 3:
            image = image.transpose(1, 2, 0)

        return image

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="images",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} images)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {self._sample_count} images, "
                    f"{throughput:.2f} images/sec"
                )
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completed: {self._sample_count} images in {elapsed:.1f}s "
                f"({throughput:.2f} images/sec)"
            )

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """Generate image for a sample."""
        features = self.qsl.get_features(sample_idx)
        prompt = features['prompt']

        # Encode prompt
        prompt_embeds = self._encode_prompt(prompt)

        # Also encode empty prompt for CFG
        empty_embeds = self._encode_prompt("")
        combined_embeds = np.concatenate([empty_embeds, prompt_embeds])

        # Generate initial latents
        latents = self._generate_latents()

        # Denoising loop
        if self.scheduler is not None:
            self.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = self.scheduler.timesteps.numpy()
        else:
            # Simple linear schedule
            timesteps = np.linspace(1000, 0, self.num_inference_steps)

        for t in timesteps:
            noise_pred = self._denoise_step(latents, combined_embeds, t)

            # Update latents (simplified scheduler step)
            if self.scheduler is not None:
                import torch
                latents_torch = torch.from_numpy(latents)
                noise_torch = torch.from_numpy(noise_pred)
                t_torch = torch.tensor([t])
                latents = self.scheduler.step(
                    noise_torch, t_torch, latents_torch
                ).prev_sample.numpy()
            else:
                # Simple update
                alpha = 1.0 - (t / 1000.0)
                latents = latents - alpha * noise_pred * 0.1

        # Decode to image
        image = self._decode_latents(latents)

        return image

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="SDXL Offline generation")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc="SDXL Server generation")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all generated images."""
        return self._predictions.copy()

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute CLIP score and FID."""
        predictions = self.get_predictions()

        if not predictions:
            return {
                'clip_score': 0.0,
                'fid_score': 0.0,
                'num_samples': 0
            }

        indices = sorted(predictions.keys())
        images = [predictions[idx] for idx in indices]

        return self.qsl.dataset.compute_accuracy(images, indices)

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
