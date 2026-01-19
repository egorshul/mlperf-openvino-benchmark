"""
Stable Diffusion XL System Under Test implementation.

This module provides SUT implementation for SDXL text-to-image model
for MLPerf Inference benchmark.

SDXL consists of multiple components:
- Text Encoder 1 (CLIP ViT-L/14)
- Text Encoder 2 (OpenCLIP ViT-bigG)
- UNet (denoising network)
- VAE Decoder (latent to image)
"""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from .config import BenchmarkConfig, Scenario
from ..datasets.coco_captions import COCOCaptionsQSL

logger = logging.getLogger(__name__)

# SDXL constants
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 8.0
IMAGE_SIZE = 1024
LATENT_CHANNELS = 4
LATENT_SIZE = IMAGE_SIZE // 8  # 128 for 1024 image


class SDXLPipeline:
    """
    SDXL Pipeline using OpenVINO for inference.

    This class manages the multi-model SDXL pipeline:
    - Tokenization (using HuggingFace tokenizers)
    - Text encoding (CLIP + OpenCLIP)
    - UNet denoising loop
    - VAE decoding
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        dtype: str = "FP16",
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ):
        """
        Initialize SDXL pipeline.

        Args:
            model_path: Path to SDXL models directory
            device: OpenVINO device (CPU, GPU, etc.)
            dtype: Model precision (FP16, FP32)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Models
        self._text_encoder_1 = None
        self._text_encoder_2 = None
        self._unet = None
        self._vae_decoder = None

        # Tokenizers
        self._tokenizer_1 = None
        self._tokenizer_2 = None

        # Scheduler
        self._scheduler = None

        self._loaded = False

    def load(self) -> None:
        """Load all SDXL models and tokenizers."""
        if self._loaded:
            return

        logger.info(f"Loading SDXL pipeline from {self.model_path}")

        try:
            import openvino as ov
            self._core = ov.Core()
        except ImportError:
            raise ImportError("OpenVINO is required. Install with: pip install openvino")

        # Load tokenizers
        self._load_tokenizers()

        # Load models
        self._load_text_encoders()
        self._load_unet()
        self._load_vae_decoder()

        # Initialize scheduler
        self._init_scheduler()

        self._loaded = True
        logger.info("SDXL pipeline loaded successfully")

    def _load_tokenizers(self) -> None:
        """Load CLIP and OpenCLIP tokenizers."""
        try:
            from transformers import CLIPTokenizer
        except ImportError:
            raise ImportError("transformers is required. Install with: pip install transformers")

        # Tokenizer 1 (CLIP)
        tokenizer_1_path = self.model_path / "tokenizer"
        if tokenizer_1_path.exists():
            self._tokenizer_1 = CLIPTokenizer.from_pretrained(str(tokenizer_1_path))
        else:
            logger.info("Loading CLIP tokenizer from HuggingFace...")
            self._tokenizer_1 = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        # Tokenizer 2 (OpenCLIP)
        tokenizer_2_path = self.model_path / "tokenizer_2"
        if tokenizer_2_path.exists():
            self._tokenizer_2 = CLIPTokenizer.from_pretrained(str(tokenizer_2_path))
        else:
            logger.info("Loading OpenCLIP tokenizer from HuggingFace...")
            self._tokenizer_2 = CLIPTokenizer.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            )

        logger.info("Tokenizers loaded")

    def _load_text_encoders(self) -> None:
        """Load text encoder models."""
        import openvino as ov

        # Text Encoder 1
        te1_path = self.model_path / "text_encoder" / "openvino_model.xml"
        if not te1_path.exists():
            te1_path = self.model_path / "text_encoder.xml"

        if te1_path.exists():
            logger.info(f"Loading Text Encoder 1 from {te1_path}")
            model = self._core.read_model(str(te1_path))
            self._text_encoder_1 = self._core.compile_model(model, self.device)
        else:
            logger.warning(f"Text Encoder 1 not found at {te1_path}")

        # Text Encoder 2
        te2_path = self.model_path / "text_encoder_2" / "openvino_model.xml"
        if not te2_path.exists():
            te2_path = self.model_path / "text_encoder_2.xml"

        if te2_path.exists():
            logger.info(f"Loading Text Encoder 2 from {te2_path}")
            model = self._core.read_model(str(te2_path))
            self._text_encoder_2 = self._core.compile_model(model, self.device)
        else:
            logger.warning(f"Text Encoder 2 not found at {te2_path}")

    def _load_unet(self) -> None:
        """Load UNet model."""
        import openvino as ov

        unet_path = self.model_path / "unet" / "openvino_model.xml"
        if not unet_path.exists():
            unet_path = self.model_path / "unet.xml"

        if unet_path.exists():
            logger.info(f"Loading UNet from {unet_path}")
            model = self._core.read_model(str(unet_path))

            # Set performance hints for UNet (most compute-intensive)
            config = {
                "PERFORMANCE_HINT": "THROUGHPUT",
            }
            self._unet = self._core.compile_model(model, self.device, config)
        else:
            raise FileNotFoundError(f"UNet model not found at {unet_path}")

    def _load_vae_decoder(self) -> None:
        """Load VAE decoder model."""
        import openvino as ov

        vae_path = self.model_path / "vae_decoder" / "openvino_model.xml"
        if not vae_path.exists():
            vae_path = self.model_path / "vae_decoder.xml"

        if vae_path.exists():
            logger.info(f"Loading VAE Decoder from {vae_path}")
            model = self._core.read_model(str(vae_path))
            self._vae_decoder = self._core.compile_model(model, self.device)
        else:
            raise FileNotFoundError(f"VAE Decoder not found at {vae_path}")

    def _init_scheduler(self) -> None:
        """Initialize the diffusion scheduler."""
        try:
            from diffusers import EulerDiscreteScheduler
            self._scheduler = EulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="scheduler"
            )
        except Exception as e:
            logger.warning(f"Could not load scheduler from HuggingFace: {e}")
            # Will use a simple scheduler implementation
            self._scheduler = None

    def tokenize(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize a prompt using both tokenizers.

        Args:
            prompt: Text prompt

        Returns:
            Tuple of (tokens_1, tokens_2)
        """
        # Tokenize with first tokenizer (CLIP)
        tokens_1 = self._tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self._tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="np"
        )

        # Tokenize with second tokenizer (OpenCLIP)
        tokens_2 = self._tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self._tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="np"
        )

        return tokens_1["input_ids"], tokens_2["input_ids"]

    def encode_prompt(
        self,
        tokens_1: np.ndarray,
        tokens_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode tokenized prompts using text encoders.

        Args:
            tokens_1: Tokens from first tokenizer
            tokens_2: Tokens from second tokenizer

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        # Encode with Text Encoder 1
        if self._text_encoder_1 is not None:
            te1_output = self._text_encoder_1(tokens_1)
            prompt_embeds_1 = te1_output[0]  # Hidden states
        else:
            # Placeholder
            prompt_embeds_1 = np.zeros((1, 77, 768), dtype=np.float32)

        # Encode with Text Encoder 2
        if self._text_encoder_2 is not None:
            te2_output = self._text_encoder_2(tokens_2)
            prompt_embeds_2 = te2_output[0]  # Hidden states
            pooled_prompt_embeds = te2_output[1] if len(te2_output) > 1 else te2_output[0][:, 0]
        else:
            # Placeholder
            prompt_embeds_2 = np.zeros((1, 77, 1280), dtype=np.float32)
            pooled_prompt_embeds = np.zeros((1, 1280), dtype=np.float32)

        # Concatenate embeddings
        prompt_embeds = np.concatenate([prompt_embeds_1, prompt_embeds_2], axis=-1)

        return prompt_embeds, pooled_prompt_embeds

    def generate(
        self,
        prompt: str,
        latent: Optional[np.ndarray] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt
            latent: Pre-generated latent (for determinism)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale

        Returns:
            Generated image as numpy array [H, W, C]
        """
        if not self._loaded:
            self.load()

        num_steps = num_inference_steps or self.num_inference_steps
        cfg_scale = guidance_scale or self.guidance_scale

        # Tokenize prompt
        tokens_1, tokens_2 = self.tokenize(prompt)

        # Encode prompt
        prompt_embeds, pooled_embeds = self.encode_prompt(tokens_1, tokens_2)

        # For CFG, we need unconditional embeddings too
        uncond_tokens_1, uncond_tokens_2 = self.tokenize("")
        uncond_embeds, uncond_pooled = self.encode_prompt(uncond_tokens_1, uncond_tokens_2)

        # Concatenate for batch processing
        text_embeds = np.concatenate([uncond_embeds, prompt_embeds], axis=0)
        pooled_embeds = np.concatenate([uncond_pooled, pooled_embeds], axis=0)

        # Initialize or use provided latent
        if latent is None:
            latent = np.random.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
            latent = latent.astype(np.float32)

        # Ensure latent has batch dimension
        if latent.ndim == 3:
            latent = np.expand_dims(latent, 0)

        # Get timesteps
        if self._scheduler is not None:
            self._scheduler.set_timesteps(num_steps)
            timesteps = self._scheduler.timesteps.numpy()
        else:
            # Simple linear schedule
            timesteps = np.linspace(999, 0, num_steps, dtype=np.int64)

        # Denoising loop
        for t in timesteps:
            # Expand latent for CFG
            latent_input = np.concatenate([latent, latent], axis=0)

            # Prepare timestep
            timestep = np.array([t, t], dtype=np.int64)

            # UNet prediction
            noise_pred = self._unet_forward(
                latent_input, timestep, text_embeds, pooled_embeds
            )

            # CFG
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            if self._scheduler is not None:
                import torch
                latent_torch = torch.from_numpy(latent)
                noise_torch = torch.from_numpy(noise_pred)
                latent_torch = self._scheduler.step(noise_torch, t, latent_torch).prev_sample
                latent = latent_torch.numpy()
            else:
                # Simple Euler step
                latent = latent - noise_pred * (1.0 / num_steps)

        # VAE decode
        image = self._vae_decode(latent)

        return image

    def _unet_forward(
        self,
        latent: np.ndarray,
        timestep: np.ndarray,
        text_embeds: np.ndarray,
        pooled_embeds: np.ndarray,
    ) -> np.ndarray:
        """Run UNet forward pass."""
        # Prepare inputs based on model's expected names
        # SDXL UNet typically expects: sample, timestep, encoder_hidden_states, text_embeds, time_ids
        inputs = {
            "sample": latent.astype(np.float32),
            "timestep": timestep,
            "encoder_hidden_states": text_embeds.astype(np.float32),
        }

        # Add additional inputs if model expects them
        try:
            input_names = [inp.any_name for inp in self._unet.inputs]

            if "text_embeds" in input_names or "added_cond_kwargs.text_embeds" in input_names:
                inputs["text_embeds"] = pooled_embeds.astype(np.float32)

            if "time_ids" in input_names or "added_cond_kwargs.time_ids" in input_names:
                # Default time_ids for SDXL: [height, width, 0, 0, height, width]
                time_ids = np.array([[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=np.float32)
                inputs["time_ids"] = time_ids
        except Exception as e:
            logger.debug(f"Could not determine UNet input names: {e}")

        output = self._unet(inputs)
        return output[0]

    def _vae_decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent to image using VAE."""
        # Scale latent
        latent = latent / 0.13025  # SDXL VAE scaling factor

        output = self._vae_decoder(latent.astype(np.float32))
        image = output[0]

        # Post-process: [1, C, H, W] -> [H, W, C], scale to [0, 255]
        image = np.squeeze(image, 0)  # Remove batch
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        image = (image / 2 + 0.5).clip(0, 1)  # Scale to [0, 1]
        image = (image * 255).astype(np.uint8)

        return image


class SDXLSUT:
    """
    System Under Test for SDXL text-to-image model.

    This SUT handles:
    - Prompt processing
    - Image generation via SDXL pipeline
    - LoadGen integration
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        pipeline: SDXLPipeline,
        qsl: COCOCaptionsQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize SDXL SUT.

        Args:
            config: Benchmark configuration
            pipeline: SDXL pipeline instance
            qsl: Query Sample Library
            scenario: MLPerf scenario
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        self.config = config
        self.pipeline = pipeline
        self.qsl = qsl
        self.scenario = scenario

        # Results storage
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0

        # LoadGen handles
        self._sut = None
        self._qsl_handle = None

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """
        Process a single sample.

        Args:
            sample_idx: Sample index

        Returns:
            Generated image
        """
        features = self.qsl.get_features(sample_idx)
        prompt = features["prompt"]
        latent = features.get("latent")

        # Generate image
        image = self.pipeline.generate(prompt=prompt, latent=latent)

        return image

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries in Offline mode."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_samples, desc="SDXL Offline", file=sys.stderr)
        else:
            pbar = None
            logger.info(f"Processing {total_samples} samples...")

        for qs in query_samples:
            sample_idx = qs.index

            # Generate image
            image = self._process_sample(sample_idx)

            # Store prediction
            self._predictions[sample_idx] = image

            # Create response (just acknowledge completion)
            response_data = np.array([1], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            responses.append(response)

            self._sample_count += 1
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries in Server mode."""
        responses = []
        response_arrays = []

        for qs in query_samples:
            sample_idx = qs.index

            # Generate image
            image = self._process_sample(sample_idx)

            # Store prediction
            self._predictions[sample_idx] = image

            # Create response
            response_data = np.array([1], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(qs.id, bi[0], bi[1])
            responses.append(response)

            self._sample_count += 1

        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        pass

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut is None:
            self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    @property
    def name(self) -> str:
        """Get SUT name."""
        return f"SDXL-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all generated images."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset SUT state."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy metrics (FID and CLIP score).

        Returns:
            Dictionary with FID and CLIP scores
        """
        if not self._predictions:
            logger.warning("No predictions to compute accuracy")
            return {"fid": 0.0, "clip_score": 0.0, "num_samples": 0}

        # Get generated images and prompts
        images = []
        prompts = []
        for idx in sorted(self._predictions.keys()):
            images.append(self._predictions[idx])
            prompts.append(self.qsl.get_label(idx))

        # Compute metrics
        try:
            from ..datasets.sdxl_metrics import compute_fid_clip_scores
            return compute_fid_clip_scores(
                images,
                prompts,
                self.qsl.dataset.data_path
            )
        except ImportError:
            logger.warning("SDXL metrics module not available")
            return {
                "fid": 0.0,
                "clip_score": 0.0,
                "num_samples": len(images),
            }
