"""Stable Diffusion XL System Under Test (OpenVINO GenAI).

Each die runs in a separate process.  In Offline mode all workers share
a single input queue so a faster die naturally picks up more work
(work-stealing).  Server mode uses a single in-process pipeline.
"""

import array
import logging
import multiprocessing
import queue
import re
import sys
import time
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
    import openvino_genai as ov_genai
    GENAI_SDXL_AVAILABLE = True
except ImportError:
    GENAI_SDXL_AVAILABLE = False
    ov_genai = None

from .config import BenchmarkConfig, Scenario
from ..datasets.coco_prompts import COCOPromptsQSL

logger = logging.getLogger(__name__)

DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)


# ------------------------------------------------------------------
# Worker process entry point
# ------------------------------------------------------------------

def _sdxl_die_worker_fn(
    die_name: str,
    model_path: str,
    image_size: int,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: str,
    input_queue,
    output_queue,
    ready_event,
) -> None:
    """Worker: load Text2ImagePipeline on *die_name*, then process samples."""
    import openvino_genai as _ov_genai
    from pathlib import Path as _Path

    pipe = _ov_genai.Text2ImagePipeline(model_path)

    scheduler_cfg = _Path(model_path) / "scheduler" / "scheduler_config.json"
    if scheduler_cfg.exists():
        scheduler = _ov_genai.Scheduler.from_config(
            str(scheduler_cfg),
            _ov_genai.Scheduler.Type.EULER_DISCRETE,
        )
        pipe.set_scheduler(scheduler)

    pipe.reshape(1, image_size, image_size, guidance_scale)

    # After reshape() all models have fully static shapes —
    # compile everything on the target device.
    pipe.compile(die_name, die_name, die_name)

    # Warmup (1 step to trigger compilation)
    gen = _ov_genai.TorchGenerator(0)
    pipe.generate(
        "warmup",
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=1,
        width=image_size,
        height=image_size,
        generator=gen,
    )

    ready_event.set()

    while True:
        item = input_queue.get()
        if item is None:
            break

        item_idx, sample_id, prompt = item

        generator = _ov_genai.TorchGenerator(0)
        image_tensor = pipe.generate(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=image_size,
            height=image_size,
            generator=generator,
        )

        image = np.array(image_tensor.data[0], copy=False)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        output_queue.put((item_idx, sample_id, image.tobytes(), image.shape))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# SUT
# ------------------------------------------------------------------

class SDXLMultiDieSUT:
    """SDXL text-to-image SUT (OpenVINO GenAI).

    Offline + multiple dies  → one worker *process* per die, shared queue.
    Offline + single die     → in-process, sequential.
    Server (any die count)   → in-process, sequential.
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
        if not GENAI_SDXL_AVAILABLE:
            raise ImportError(
                "openvino-genai is required for SDXL GenAI inference. "
                "Install with: pip install openvino-genai"
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
        self._completed = 0
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        # Multi-process state (Offline, >1 die)
        self._workers: List[Tuple[str, multiprocessing.Process]] = []
        self._input_queue = None
        self._output_queue = None

        # In-process state (single die / Server)
        self._pipeline = None

        self._setup()

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_device_dies(device: str) -> List[str]:
        """Return sorted list of sub-device identifiers (e.g. NPU.0, NPU.1)."""
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        target_device = (
            self.config.openvino.device
            if hasattr(self.config, "openvino")
            else "CPU"
        )

        if target_device == "CPU":
            device_dies = ["CPU"]
        elif "," in target_device:
            device_dies = [p.strip() for p in target_device.split(",")]
        elif re.match(r"^.+\.\d+$", target_device):
            device_dies = [target_device]
        else:
            device_dies = self._discover_device_dies(target_device)
            if not device_dies:
                logger.warning("No %s dies found, using single device", target_device)
                device_dies = [target_device]

        if self.scenario == Scenario.OFFLINE and len(device_dies) > 1:
            self._setup_workers(device_dies)
        else:
            self._setup_pipeline_inprocess(device_dies[0])

    def _setup_workers(self, device_dies: List[str]) -> None:
        """Spawn one worker process per die with a shared input queue."""
        ctx = multiprocessing.get_context("spawn")
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()

        logger.info(f"[SDXL] Spawning {len(device_dies)} worker process(es)...")

        pending: List[Tuple[str, multiprocessing.Process, Any]] = []

        for die_name in device_dies:
            ready = ctx.Event()
            p = ctx.Process(
                target=_sdxl_die_worker_fn,
                args=(
                    die_name,
                    str(self.model_path),
                    self.image_size,
                    self.guidance_scale,
                    self.num_inference_steps,
                    self.negative_prompt,
                    self._input_queue,
                    self._output_queue,
                    ready,
                ),
                daemon=True,
            )
            p.start()
            pending.append((die_name, p, ready))
            logger.info(f"  {die_name}: spawned (pid={p.pid})")

        for die_name, p, ready in pending:
            if not ready.wait(timeout=600):
                raise RuntimeError(
                    f"Worker for {die_name} did not become ready in 600 s"
                )
            self._workers.append((die_name, p))
            logger.info(f"  {die_name}: compiled & warmed up")

        logger.info(f"[SDXL] {len(self._workers)} worker(s) ready")

    def _setup_pipeline_inprocess(self, die: str) -> None:
        """Create and compile a single Text2ImagePipeline in the current process."""
        print(f"[SDXL] Compiling on {die} ...", file=sys.stderr, flush=True)

        pipe = ov_genai.Text2ImagePipeline(str(self.model_path))

        scheduler_cfg = self.model_path / "scheduler" / "scheduler_config.json"
        if scheduler_cfg.exists():
            scheduler = ov_genai.Scheduler.from_config(
                str(scheduler_cfg),
                ov_genai.Scheduler.Type.EULER_DISCRETE,
            )
            pipe.set_scheduler(scheduler)

        pipe.reshape(1, self.image_size, self.image_size, self.guidance_scale)

        # After reshape() all models have fully static shapes —
        # compile everything on the target device.
        pipe.compile(die, die, die)

        self._pipeline = pipe
        print(f"[SDXL] 1 die: {die}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _process_sample(self, sample_idx: int) -> np.ndarray:
        """Generate one image with the in-process pipeline."""
        features = self.qsl.get_features(sample_idx)
        prompt = features["prompt"]

        generator = ov_genai.TorchGenerator(0)
        image_tensor = self._pipeline.generate(
            prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            width=self.image_size,
            height=self.image_size,
            generator=generator,
        )

        image = np.array(image_tensor.data[0], copy=False)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    # ------------------------------------------------------------------
    # Query dispatch
    # ------------------------------------------------------------------

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        self._start_time = time.time()
        self._completed = 0

        if self._workers:
            print(
                f"[Offline] {total} samples, {len(self._workers)} worker process(es)",
                file=sys.stderr,
            )
            self._issue_offline_multiprocess(query_samples)
        else:
            print(f"[Offline] {total} samples, 1 die (in-process)", file=sys.stderr)
            self._issue_offline_sequential(query_samples)

    def _issue_offline_multiprocess(self, query_samples: List[Any]) -> None:
        total = len(query_samples)

        # Enqueue all samples into the shared queue
        for i, sample in enumerate(query_samples):
            prompt = self.qsl.get_features(sample.index)["prompt"]
            self._input_queue.put((i, sample.id, prompt))

        # Collect results
        collected: Dict[int, Tuple] = {}

        while len(collected) < total:
            # Check worker health
            for die_name, p in self._workers:
                if not p.is_alive():
                    raise RuntimeError(
                        f"Worker {die_name} died (exit code {p.exitcode})"
                    )

            # Drain output queue
            try:
                while True:
                    result = self._output_queue.get_nowait()
                    collected[result[0]] = result
            except queue.Empty:
                pass

            self._completed = len(collected)
            _print_progress(self._completed, total, self._start_time)

            if len(collected) < total:
                time.sleep(0.5)

        _print_progress(total, total, self._start_time)

        # Build LoadGen responses in original order
        responses = []
        arrays = []
        for i, sample in enumerate(query_samples):
            _, _, image_bytes, image_shape = collected[i]
            image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)

            self._predictions[sample.index] = image

            response_data = np.array(image_shape, dtype=np.int64)
            arr = array.array("B", response_data.tobytes())
            arrays.append(arr)
            bi = arr.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)

    def _issue_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        arrays = []

        for sample in query_samples:
            image = self._process_sample(sample.index)

            self._predictions[sample.index] = image
            self._completed += 1

            response_data = np.array(image.shape, dtype=np.int64)
            arr = array.array("B", response_data.tobytes())
            arrays.append(arr)
            bi = arr.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
            _print_progress(self._completed, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            image = self._process_sample(sample.index)

            self._predictions[sample.index] = image
            self._completed += 1

            response_data = np.array(image.shape, dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

    # ------------------------------------------------------------------
    # LoadGen interface
    # ------------------------------------------------------------------

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

    def shutdown(self) -> None:
        """Send poison pills and join worker processes."""
        for _ in self._workers:
            try:
                self._input_queue.put_nowait(None)
            except Exception:
                pass
        for _, p in self._workers:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
