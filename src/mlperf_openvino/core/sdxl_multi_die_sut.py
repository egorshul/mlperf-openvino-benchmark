"""Stable Diffusion XL Multi-Die System Under Test."""

import array
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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


class SDXLMultiDieSUT:
    """SDXL text-to-image SUT for multi-die accelerators.

    Loads one OVStableDiffusionXLPipeline per die. Offline mode distributes
    samples across dies in parallel; Server mode uses round-robin dispatch.
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
        self._lock = threading.Lock()
        self._completed = 0
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._pipelines: List[Tuple[str, Any, threading.Lock]] = []
        self._pipeline_index = 0

        self._setup_pipelines()

    def _discover_device_dies(self, device: str) -> List[str]:
        """Return sorted list of sub-device identifiers (e.g. NPU.0, NPU.1)."""
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup_pipelines(self) -> None:
        target_device = (
            self.config.openvino.device
            if hasattr(self.config, "openvino")
            else "CPU"
        )

        if target_device == "CPU":
            device_dies = ["CPU"]
        elif re.match(r"^.+\.\d+$", target_device):
            device_dies = [target_device]
        else:
            device_dies = self._discover_device_dies(target_device)
            if not device_dies:
                logger.warning("No %s dies found, using single device", target_device)
                device_dies = [target_device]

        for die in device_dies:
            try:
                print(f"[SDXL] Compiling on {die} ...", file=sys.stderr, flush=True)
                pipeline = self._load_pipeline_for_device(die)
                self._pipelines.append((die, pipeline, threading.Lock()))
            except Exception as exc:
                logger.debug("Failed to load pipeline for %s: %s", die, exc)

        if not self._pipelines:
            logger.warning("No accelerator pipelines loaded, falling back to CPU")
            pipeline = self._load_pipeline_for_device("CPU")
            self._pipelines.append(("CPU", pipeline, threading.Lock()))

        die_names = [name for name, _, _ in self._pipelines]
        bs_info = f", batch={self.batch_size}" if self.batch_size > 1 else ""
        print(
            f"[SDXL] {len(self._pipelines)} die(s): {', '.join(die_names)}{bs_info}",
            file=sys.stderr,
        )

    def _load_pipeline_for_device(self, die: str) -> Any:
        is_cpu = die.upper() == "CPU"

        if is_cpu and self.batch_size <= 1:
            pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                str(self.model_path), compile=True, load_in_8bit=False,
            )
        else:
            pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                str(self.model_path), compile=False, load_in_8bit=False,
            )
            try:
                pipeline.reshape(
                    batch_size=self.batch_size,
                    height=self.image_size,
                    width=self.image_size,
                    num_images_per_prompt=1,
                )
                if not is_cpu:
                    pipeline.to(die)
                pipeline.compile()
            except Exception as exc:
                if self.batch_size > 1:
                    logger.warning(
                        "batch=%d failed on %s (%s), falling back to 1",
                        self.batch_size, die, exc,
                    )
                    pipeline = OVStableDiffusionXLPipeline.from_pretrained(
                        str(self.model_path), compile=False, load_in_8bit=False,
                    )
                    pipeline.reshape(
                        batch_size=1,
                        height=self.image_size,
                        width=self.image_size,
                        num_images_per_prompt=1,
                    )
                    if not is_cpu:
                        pipeline.to(die)
                    pipeline.compile()
                    self.batch_size = 1
                else:
                    raise

        pipeline.set_progress_bar_config(disable=True)

        try:
            from diffusers import EulerDiscreteScheduler
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing="leading",
                steps_offset=1,
                prediction_type="epsilon",
            )
        except Exception:
            logger.warning("Failed to set EulerDiscreteScheduler on %s", die)

        if hasattr(pipeline, "watermark"):
            pipeline.watermark = None

        return pipeline

    def _process_sample(self, sample_idx: int, pipeline: Any) -> np.ndarray:
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

        image = pipeline(**pipe_kwargs).images[0]

        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).round().astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
        else:
            image = np.array(image)

        return image

    def _process_batch(self, sample_indices: List[int], pipeline: Any) -> List[np.ndarray]:
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

        result = pipeline(**pipe_kwargs)

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

    @property
    def _sample_count(self) -> int:
        # Exposes completed count under the name expected by benchmark_runner.
        return self._completed

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        num_dies = len(self._pipelines)
        self._start_time = time.time()
        self._completed = 0

        print(f"[Offline] {total} samples, {num_dies} die(s)", file=sys.stderr)

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        die_batches: List[List[Tuple[Any, int]]] = [[] for _ in range(num_dies)]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        all_results: List[Tuple[Any, int, np.ndarray]] = []
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]):
            _name, pipeline, die_lock = self._pipelines[die_idx]
            bs = self.batch_size
            for i in range(0, len(batch), bs):
                chunk = batch[i:i + bs]
                indices = [idx for _, idx in chunk]

                if len(indices) < bs:
                    indices_padded = indices + [indices[-1]] * (bs - len(indices))
                else:
                    indices_padded = indices

                with die_lock:
                    if bs > 1:
                        images = self._process_batch(indices_padded, pipeline)
                        images = images[:len(chunk)]
                    else:
                        images = [self._process_sample(indices[0], pipeline)]

                for (sample, sample_idx), image in zip(chunk, images):
                    with self._lock:
                        self._predictions[sample_idx] = image
                        self._completed += 1
                    with results_lock:
                        all_results.append((sample, sample_idx, image))

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = [
                pool.submit(_die_worker, idx, batch)
                for idx, batch in enumerate(die_batches)
                if batch
            ]
            while True:
                done = sum(1 for f in futures if f.done())
                with self._lock:
                    completed = self._completed
                _print_progress(completed, total, self._start_time)
                if done == len(futures):
                    break
                time.sleep(0.5)

            for f in futures:
                f.result()

        _print_progress(total, total, self._start_time)
        self._send_loadgen_responses(all_results)

    def _issue_queries_offline_sequential(
        self, query_samples: List[Any],
    ) -> None:
        total = len(query_samples)
        all_results: List[Tuple[Any, int, np.ndarray]] = []

        _, pipeline, _ = self._pipelines[0]
        bs = self.batch_size
        for i in range(0, total, bs):
            chunk = query_samples[i:i + bs]
            indices = [s.index for s in chunk]

            if len(indices) < bs:
                indices_padded = indices + [indices[-1]] * (bs - len(indices))
            else:
                indices_padded = indices

            if bs > 1:
                images = self._process_batch(indices_padded, pipeline)
                images = images[:len(chunk)]
            else:
                images = [self._process_sample(indices[0], pipeline)]

            for sample, image in zip(chunk, images):
                self._predictions[sample.index] = image
                self._completed += 1
                all_results.append((sample, sample.index, image))
            _print_progress(self._completed, total, self._start_time)

        _print_progress(total, total, self._start_time)
        self._send_loadgen_responses(all_results)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            _name, pipeline, die_lock = self._pipelines[self._pipeline_index]
            self._pipeline_index = (self._pipeline_index + 1) % len(self._pipelines)

            with die_lock:
                image = self._process_sample(sample_idx, pipeline)

            with self._lock:
                self._predictions[sample_idx] = image
                self._completed += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

    @staticmethod
    def _send_loadgen_responses(
        results: List[Tuple[Any, int, np.ndarray]],
    ) -> None:
        responses = []
        # array.array objects must stay alive until QuerySamplesComplete returns.
        arrays = []
        for sample, _idx, image in sorted(results, key=lambda r: r[1]):
            response_data = np.array(image.shape, dtype=np.int64)
            arr = array.array("B", response_data.tobytes())
            arrays.append(arr)
            bi = arr.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)

    def flush_queries(self) -> None:
        pass

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries, self.flush_queries,
            )
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
        self._completed = 0
        self._query_count = 0
        self._pipeline_index = 0
