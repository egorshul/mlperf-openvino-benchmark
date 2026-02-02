"""
Stable Diffusion XL Multi-Die System Under Test implementation.

Loads one OVStableDiffusionXLPipeline per accelerator die and
distributes image generation across dies in parallel (Offline)
or round-robin (Server).
"""

import array
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

from .config import BenchmarkConfig, Scenario
from ..datasets.coco_prompts import COCOPromptsQSL

logger = logging.getLogger(__name__)

# SDXL default parameters (MLCommons reference implementation)
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"


class SDXLMultiDieSUT:
    """
    System Under Test for SDXL on multi-die accelerators.

    Loads one OVStableDiffusionXLPipeline per accelerator die and
    distributes image generation across dies in parallel (Offline)
    or round-robin (Server).

    Python-only implementation.
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
        """Initialize SDXL multi-die SUT."""
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_SDXL_AVAILABLE:
            raise ImportError(
                "Optimum-Intel with SDXL support is required for multi-die SUT. "
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

        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        # List of (die_name, pipeline) tuples
        self._pipelines: List[Tuple[str, Any]] = []
        self._pipeline_index = 0  # Round-robin index for Server mode

        self._setup_pipelines()

    def _discover_device_dies(self, device: str) -> List[str]:
        """Discover available dies for the given device prefix."""
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")

        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        device_dies = [d for d in devices if pattern.match(d)]
        return sorted(device_dies)

    def _setup_pipelines(self) -> None:
        """Load one OVStableDiffusionXLPipeline per accelerator die."""
        target_device = (
            self.config.openvino.device
            if hasattr(self.config, 'openvino')
            else "CPU"
        )

        logger.info(f"Setting up SDXL Multi-Die SUT (device={target_device})")

        # Determine dies to use
        if target_device == "CPU":
            device_dies = ["CPU"]
        elif re.match(r"^.+\.\d+$", target_device):
            # Specific die requested (e.g., NPU.0)
            device_dies = [target_device]
        else:
            # Discover all dies for device prefix (e.g., NPU -> [NPU.0, NPU.1])
            device_dies = self._discover_device_dies(target_device)
            if not device_dies:
                logger.warning(
                    f"No {target_device} dies found, falling back to single device"
                )
                device_dies = [target_device]

        # Load a pipeline per die
        for die in device_dies:
            logger.info(f"Loading OVStableDiffusionXLPipeline for {die}")
            try:
                pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path),
                    device=die,
                    compile=True,
                    load_in_8bit=False,
                )

                # Override scheduler to EulerDiscreteScheduler (MLCommons requirement)
                try:
                    from diffusers import EulerDiscreteScheduler
                    pipeline.scheduler = EulerDiscreteScheduler.from_config(
                        pipeline.scheduler.config
                    )
                except Exception as e:
                    logger.warning(f"Failed to override scheduler on {die}: {e}")

                self._pipelines.append((die, pipeline))
                logger.info(f"Pipeline loaded for {die}")

            except Exception as e:
                logger.warning(f"Failed to load pipeline for {die}: {e}")

        if not self._pipelines:
            # Fallback: load on CPU
            logger.warning("No accelerator pipelines loaded, falling back to CPU")
            pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                str(self.model_path),
                compile=True,
                load_in_8bit=False,
            )
            try:
                from diffusers import EulerDiscreteScheduler
                pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    pipeline.scheduler.config
                )
            except Exception:
                pass
            self._pipelines.append(("CPU", pipeline))

        die_names = [die for die, _ in self._pipelines]
        print(
            f"[Setup] SDXL Multi-Die: {len(self._pipelines)} die(s) = {die_names}",
            file=sys.stderr,
        )

    def _get_next_pipeline(self) -> Tuple[str, Any]:
        """Get next pipeline in round-robin order (for Server mode)."""
        die_name, pipeline = self._pipelines[self._pipeline_index]
        self._pipeline_index = (self._pipeline_index + 1) % len(self._pipelines)
        return die_name, pipeline

    def _process_sample(self, sample_idx: int, pipeline: Any) -> np.ndarray:
        """Generate an image for the given sample using the specified pipeline.

        MLCommons compliance: pre-computed latents, EulerDiscreteScheduler,
        20 steps, guidance_scale=8.0, specific negative prompt.
        """
        features = self.qsl.get_features(sample_idx)
        prompt = features['prompt']

        guidance_scale = features.get('guidance_scale', self.guidance_scale)
        num_steps = features.get('num_inference_steps', self.num_inference_steps)

        # Use pre-computed latents if available (REQUIRED for closed division)
        latents = features.get('latents', None)

        if latents is not None:
            try:
                import torch
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents.copy()).float()
                if latents.dim() == 3:
                    latents = latents.unsqueeze(0)
                if latents.shape[1] != 4:
                    logger.warning(
                        f"Latents have {latents.shape[1]} channels, expected 4. "
                        "Using random latents instead."
                    )
                    latents = None
            except ImportError:
                logger.warning("PyTorch not available, cannot convert latents")
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
            logger.warning(
                f"No pre-computed latents for sample {sample_idx}. "
                "Results may not match MLCommons reference."
            )
            try:
                import torch
                pipe_kwargs["generator"] = torch.Generator().manual_seed(sample_idx)
            except ImportError:
                pass

        result = pipeline(**pipe_kwargs)

        image = result.images[0]
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).round().astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
        else:
            image = np.array(image)

        return image

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        """Offline mode: distribute samples across dies in parallel."""
        total = len(query_samples)
        num_dies = len(self._pipelines)
        self._start_time = time.time()

        print(
            f"[Offline] SDXL: {total} samples, {num_dies} die(s)",
            file=sys.stderr,
        )

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        # Distribute samples round-robin across dies
        die_batches: List[List[Tuple[Any, int]]] = [[] for _ in range(num_dies)]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        batch_counts = [len(b) for b in die_batches]
        print(f"[Distribute] {batch_counts} samples per die", file=sys.stderr)

        def _worker(die_idx: int, batch: List[Tuple[Any, int]]):
            _, pipeline = self._pipelines[die_idx]
            results = []
            for sample, sample_idx in batch:
                image = self._process_sample(sample_idx, pipeline)
                results.append((sample, sample_idx, image))
            return results

        all_results: List[Tuple[Any, int, np.ndarray]] = []
        print("[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = {
                pool.submit(_worker, die_idx, batch): die_idx
                for die_idx, batch in enumerate(die_batches)
                if batch
            }
            for future in as_completed(futures):
                for sample, sample_idx, image in future.result():
                    self._predictions[sample_idx] = image
                    self._sample_count += 1

                    progress = int(self._sample_count * 10 / total)
                    while dots_printed < progress:
                        print(".", end="", file=sys.stderr, flush=True)
                        dots_printed += 1

                all_results.extend(future.result())

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(
            f" {total}/{total} ({elapsed:.1f}s, {total/elapsed:.1f} images/sec)",
            file=sys.stderr,
        )

        # Send responses to LoadGen
        responses = []
        response_arrays = []
        for sample, _idx, image in sorted(all_results, key=lambda r: r[1]):
            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(self, query_samples: List[Any]) -> None:
        """Single-die fallback for Offline mode."""
        total = len(query_samples)
        responses = []
        response_arrays = []

        print("[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        for sample in query_samples:
            sample_idx = sample.index
            _, pipeline = self._pipelines[0]

            image = self._process_sample(sample_idx, pipeline)
            self._predictions[sample_idx] = image
            self._sample_count += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            if total > 0:
                progress = int(self._sample_count * 10 / total)
                while dots_printed < progress:
                    print(".", end="", file=sys.stderr, flush=True)
                    dots_printed += 1

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(
            f" {total}/{total} ({elapsed:.1f}s, {total/elapsed:.1f} images/sec)",
            file=sys.stderr,
        )

        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        """Server mode: round-robin across dies, respond per-query."""
        for sample in query_samples:
            sample_idx = sample.index
            _, pipeline = self._get_next_pipeline()

            image = self._process_sample(sample_idx, pipeline)
            self._predictions[sample_idx] = image
            self._sample_count += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        pass

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries, self.flush_queries
            )
        return self._sut_handle

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
                'num_samples': 0,
            }

        indices = sorted(predictions.keys())
        images = [predictions[idx] for idx in indices]

        return self.qsl.dataset.compute_accuracy(images, indices)

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._pipeline_index = 0
