"""Stable Diffusion XL System Under Test."""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    from optimum.intel import OVStableDiffusionXLPipeline
    OPTIMUM_SDXL_AVAILABLE = True
except ImportError:
    OPTIMUM_SDXL_AVAILABLE = False
    OVStableDiffusionXLPipeline = None

try:
    from diffusers import StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionXLPipeline = None

from .config import BenchmarkConfig, Scenario
from ..datasets.coco_prompts import COCOPromptsQSL

logger = logging.getLogger(__name__)

DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _print_progress(completed: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    if completed >= total:
        line = f"\r[Inference] {completed}/{total} | {rate:.2f} samples/s | {_fmt_time(elapsed)}"
        print(line, file=sys.stderr, flush=True)
        print(file=sys.stderr)
    else:
        eta = (total - completed) / rate if rate > 0 else 0.0
        line = f"\r[Inference] {completed}/{total} | {rate:.2f} samples/s | ETA {_fmt_time(eta)}"
        print(line, end="", file=sys.stderr, flush=True)


class SDXLOptimumSUT:
    """SDXL SUT using Optimum-Intel OVStableDiffusionXLPipeline."""

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

        if scenario == Scenario.SERVER:
            self.batch_size = 1
        else:
            bs = config.openvino.batch_size if hasattr(config, 'openvino') else 0
            self.batch_size = bs if bs > 1 else 1

        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._load_pipeline()

    def _load_pipeline(self) -> None:
        device = self.config.openvino.device if hasattr(self.config, "openvino") else "CPU"
        print(f"[SDXL] Compiling on {device} ...", file=sys.stderr, flush=True)

        ov_config = {"EXECUTION_MODE_HINT": "ACCURACY"}

        try:
            if self.batch_size > 1:
                self.pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path), compile=False, load_in_8bit=False,
                    ov_config=ov_config,
                )
                try:
                    self.pipeline.reshape(
                        batch_size=self.batch_size,
                        height=self.image_size,
                        width=self.image_size,
                        num_images_per_prompt=1,
                    )
                    self.pipeline.compile()
                except Exception as exc:
                    logger.warning(
                        "batch=%d failed (%s), falling back to 1",
                        self.batch_size, exc,
                    )
                    self.pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                        str(self.model_path), compile=True, load_in_8bit=False,
                        ov_config=ov_config,
                    )
                    self.batch_size = 1
            else:
                self.pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path), compile=True, load_in_8bit=False,
                    ov_config=ov_config,
                )
        except Exception as e:
            logger.warning("Failed to load OpenVINO pipeline: %s", e)
            if DIFFUSERS_AVAILABLE:
                import torch
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path), torch_dtype=torch.float32,
                ).to("cpu")
            else:
                raise RuntimeError(f"Cannot load SDXL model: {e}")

        self.pipeline.set_progress_bar_config(disable=True)

        try:
            from diffusers import EulerDiscreteScheduler
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config,
                timestep_spacing="leading",
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=False,
            )
        except Exception as e:
            logger.warning("Failed to set EulerDiscreteScheduler: %s", e)

        if hasattr(self.pipeline, "watermark"):
            self.pipeline.watermark = None

    def flush_queries(self) -> None:
        pass

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        features = self.qsl.get_features(sample_idx)
        prompt = features["prompt"]
        guidance_scale = features.get("guidance_scale", self.guidance_scale)
        num_steps = features.get("num_inference_steps", self.num_inference_steps)
        latents = features.get("latents", None)

        if latents is not None:
            try:
                import torch
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents.copy()).float()
                if latents.dim() == 3:
                    latents = latents.unsqueeze(0)
                if latents.shape[1] != 4:
                    latents = None
            except ImportError:
                latents = None

        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "height": self.image_size,
            "width": self.image_size,
            "output_type": "np",
        }
        if latents is not None:
            pipe_kwargs["latents"] = latents
        else:
            try:
                import torch
                pipe_kwargs["generator"] = torch.Generator().manual_seed(sample_idx)
            except ImportError:
                pass

        image = self.pipeline(**pipe_kwargs).images[0]

        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).round().astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
        else:
            image = np.array(image)

        return image

    def _process_batch(self, sample_indices: List[int]) -> List[np.ndarray]:
        import torch

        prompts = []
        latents_list = []

        for idx in sample_indices:
            features = self.qsl.get_features(idx)
            prompts.append(features["prompt"])
            latent = features.get("latents", None)
            if latent is not None:
                if isinstance(latent, np.ndarray):
                    t = torch.from_numpy(latent.copy()).float()
                else:
                    t = torch.tensor(latent).float()
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                latents_list.append(t)

        pipe_kwargs = {
            "prompt": prompts,
            "negative_prompt": [self.negative_prompt] * len(prompts),
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "height": self.image_size,
            "width": self.image_size,
            "output_type": "np",
        }
        if latents_list and len(latents_list) == len(prompts):
            pipe_kwargs["latents"] = torch.cat(latents_list, dim=0)

        result = self.pipeline(**pipe_kwargs)

        images = []
        for img in result.images:
            if isinstance(img, np.ndarray):
                if img.max() <= 1.0:
                    img = (img * 255).round().astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
            else:
                img = np.array(img)
            images.append(img)

        return images

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            self._issue_query_server(query_samples)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        self._start_time = time.time()
        responses = []
        response_arrays = []
        bs = self.batch_size

        bs_info = f", batch={bs}" if bs > 1 else ""
        print(f"[Offline] {total} samples{bs_info}", file=sys.stderr)

        for i in range(0, total, bs):
            chunk = query_samples[i:i + bs]
            indices = [s.index for s in chunk]

            if len(indices) < bs:
                indices_padded = indices + [indices[-1]] * (bs - len(indices))
            else:
                indices_padded = indices

            if bs > 1:
                images = self._process_batch(indices_padded)
                images = images[:len(chunk)]
            else:
                images = [self._process_sample(indices[0])]

            for sample, image in zip(chunk, images):
                self._predictions[sample.index] = image
                self._sample_count += 1

                response_data = np.array(image.shape, dtype=np.int64)
                response_array = array.array("B", response_data.tobytes())
                response_arrays.append(response_array)
                bi = response_array.buffer_info()
                responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image
            self._sample_count += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        return self._predictions.copy()

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()
        if not predictions:
            return {"clip_score": 0.0, "fid_score": 0.0, "num_samples": 0}

        indices = sorted(predictions.keys())
        images = [predictions[idx] for idx in indices]
        return self.qsl.dataset.compute_accuracy(images, indices)

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class SDXLManualSUT:
    """SDXL SUT with manual OpenVINO component loading (UNet, VAE, text encoders)."""

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
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size
        self.negative_prompt = negative_prompt

        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self.unet = None
        self.vae_decoder = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.scheduler = None

        self._load_components()

    def _load_components(self) -> None:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("OpenVINO is required for SDXL inference")

        core = ov.Core()
        logger.debug(f"Loading SDXL components from {self.model_path}")

        ov_config = {"EXECUTION_MODE_HINT": "ACCURACY"}

        unet_path = self.model_path / "unet" / "openvino_model.xml"
        if not unet_path.exists():
            unet_path = self.model_path / "unet.xml"
        if unet_path.exists():
            logger.debug(f"Loading UNet from {unet_path}")
            self.unet = core.compile_model(str(unet_path), "CPU", ov_config)

        vae_path = self.model_path / "vae_decoder" / "openvino_model.xml"
        if not vae_path.exists():
            vae_path = self.model_path / "vae_decoder.xml"
        if vae_path.exists():
            logger.debug(f"Loading VAE decoder from {vae_path}")
            self.vae_decoder = core.compile_model(str(vae_path), "CPU", ov_config)

        text_enc_path = self.model_path / "text_encoder" / "openvino_model.xml"
        if not text_enc_path.exists():
            text_enc_path = self.model_path / "text_encoder.xml"
        if text_enc_path.exists():
            logger.debug(f"Loading text encoder from {text_enc_path}")
            self.text_encoder = core.compile_model(str(text_enc_path), "CPU", ov_config)

        text_enc2_path = self.model_path / "text_encoder_2" / "openvino_model.xml"
        if not text_enc2_path.exists():
            text_enc2_path = self.model_path / "text_encoder_2.xml"
        if text_enc2_path.exists():
            logger.debug(f"Loading text encoder 2 from {text_enc2_path}")
            self.text_encoder_2 = core.compile_model(str(text_enc2_path), "CPU", ov_config)

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

    def _encode_prompt(self, prompt: str):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np"
        )

        text_input_ids = tokens['input_ids']

        if self.text_encoder is not None:
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        else:
            prompt_embeds = np.zeros((1, 77, 768), dtype=np.float32)

        pooled_prompt_embeds = np.zeros((1, 1280), dtype=np.float32)

        if self.text_encoder_2 is not None and self.tokenizer_2 is not None:
            tokens_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np"
            )
            text_encoder_2_output = self.text_encoder_2(tokens_2['input_ids'])

            if hasattr(text_encoder_2_output, '__len__') and len(text_encoder_2_output) > 1:
                prompt_embeds_2 = text_encoder_2_output[0]
                pooled_prompt_embeds = text_encoder_2_output[1]
            else:
                prompt_embeds_2 = text_encoder_2_output[0] if hasattr(text_encoder_2_output, '__getitem__') else text_encoder_2_output
            prompt_embeds = np.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)

        return prompt_embeds, pooled_prompt_embeds

    def _generate_latents(self, batch_size: int = 1) -> np.ndarray:
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
        pooled_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        latent_input = np.concatenate([latents] * 2)

        if self.scheduler is not None:
            import torch
            latent_input_torch = torch.from_numpy(latent_input)
            t_torch = torch.tensor([timestep])
            latent_input = self.scheduler.scale_model_input(
                latent_input_torch, t_torch
            ).numpy()

        timestep_array = np.array([timestep], dtype=np.float32)

        unet_inputs = {
            'sample': latent_input,
            'timestep': timestep_array,
            'encoder_hidden_states': prompt_embeds,
        }

        if pooled_embeds is not None:
            unet_inputs['text_embeds'] = pooled_embeds
        if time_ids is not None:
            unet_inputs['time_ids'] = time_ids

        try:
            noise_pred = self.unet(unet_inputs)[0]
        except Exception:
            noise_pred = self.unet({
                'sample': latent_input,
                'timestep': timestep_array,
                'encoder_hidden_states': prompt_embeds,
            })[0]

        noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred

    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        latents = latents / 0.13025
        image = self.vae_decoder(latents)[0]
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image * 255).round().astype(np.uint8)

        if image.ndim == 4:
            image = image.transpose(0, 2, 3, 1)[0]
        elif image.ndim == 3:
            image = image.transpose(1, 2, 0)

        return image

    def flush_queries(self) -> None:
        pass

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        features = self.qsl.get_features(sample_idx)
        prompt = features['prompt']
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)
        negative_embeds, pooled_negative_embeds = self._encode_prompt(
            self.negative_prompt
        )

        combined_embeds = np.concatenate([negative_embeds, prompt_embeds])
        combined_pooled = np.concatenate([pooled_negative_embeds, pooled_prompt_embeds])

        time_ids_single = np.array(
            [self.image_size, self.image_size, 0, 0, self.image_size, self.image_size],
            dtype=np.float32
        ).reshape(1, 6)
        combined_time_ids = np.concatenate([time_ids_single, time_ids_single])

        latents = features.get('latents', None)
        if latents is not None:
            if latents.ndim == 3:
                latents = latents[np.newaxis, ...]
            latents = latents.astype(np.float32)
        else:
            logger.warning(
                f"No pre-computed latents for sample {sample_idx}. "
                "Using random latents (not MLCommons-compliant)."
            )
            latents = self._generate_latents()

        if self.scheduler is not None:
            self.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = self.scheduler.timesteps.numpy()
            init_noise_sigma = float(self.scheduler.init_noise_sigma)
            latents = latents * init_noise_sigma
        else:
            timesteps = np.linspace(1000, 0, self.num_inference_steps)

        for t in timesteps:
            noise_pred = self._denoise_step(
                latents, combined_embeds, t,
                pooled_embeds=combined_pooled,
                time_ids=combined_time_ids,
            )

            if self.scheduler is not None:
                import torch
                latents_torch = torch.from_numpy(latents)
                noise_torch = torch.from_numpy(noise_pred)
                t_torch = torch.tensor([t])
                latents = self.scheduler.step(
                    noise_torch, t_torch, latents_torch
                ).prev_sample.numpy()
            else:
                alpha = 1.0 - (t / 1000.0)
                latents = latents - alpha * noise_pred * 0.1

        image = self._decode_latents(latents)

        return image

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            self._issue_query_server(query_samples)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        self._start_time = time.time()
        responses = []
        response_arrays = []

        print(f"[Offline] {total} samples", file=sys.stderr)

        for sample in query_samples:
            sample_idx = sample.index
            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image
            self._sample_count += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            image = self._process_sample(sample_idx)
            self._predictions[sample_idx] = image
            self._sample_count += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        return self._predictions.copy()

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()
        if not predictions:
            return {"clip_score": 0.0, "fid_score": 0.0, "num_samples": 0}

        indices = sorted(predictions.keys())
        images = [predictions[idx] for idx in indices]
        return self.qsl.dataset.compute_accuracy(images, indices)

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
