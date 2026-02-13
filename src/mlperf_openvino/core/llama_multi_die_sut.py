"""Llama multi-die SUT for accelerator inference (NPU/VPU/XPU).

Uses openvino_genai.LLMPipeline which handles static-shape compilation,
KV-cache management, and prefill/decode switching internally — required
for devices that only support dynamic batch dimension.

Distributes text generation across multiple accelerator dies;
samples are dispatched round-robin across dies.

Follows the MLCommons Inference v5.1 reference:
  https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b
"""

import array
import logging
import re
import sys
import threading
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

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    ov_genai = None

from .config import BenchmarkConfig, Scenario
from ..datasets.cnn_dailymail import CnnDailyMailQSL

logger = logging.getLogger(__name__)


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


def _make_response(sample_id: int, token_ids: np.ndarray, n_tokens: int):
    """Build a QuerySampleResponse with int64 token IDs.

    The data is stored as int64 bytes so that the official MLCommons
    evaluation.py can parse it via np.frombuffer(..., np.int64).

    Returns:
        (response, response_array) — caller must keep response_array alive
        until QuerySamplesComplete processes the response buffer.
    """
    token_bytes = token_ids.astype(np.int64).tobytes()
    response_array = array.array("B", token_bytes)
    bi = response_array.buffer_info()
    try:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1], n_tokens)
    except TypeError:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
    return resp, response_array


class LlamaMultiDieSUT:
    """Llama multi-die SUT using OpenVINO GenAI (LLMPipeline).

    Per die: an LLMPipeline is created which handles model compilation
    with static shapes, KV-cache management, and token generation.
    Offline mode distributes samples across dies in parallel via round-robin.

    Per MLPerf v5.1 Llama 3.1 8B spec:
      - Task: text summarization (CNN-DailyMail)
      - max_new_tokens: 128
      - Greedy decoding (do_sample=False)
      - n_tokens reported per response (use_token_latencies=1)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: CnnDailyMailQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 128,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")
        if not GENAI_AVAILABLE:
            raise ImportError(
                "openvino-genai is required for LlamaMultiDieSUT. "
                "Install with: pip install openvino-genai"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        self._store_predictions = True
        self._predictions: Dict[int, str] = {}
        self._predictions_lock = threading.Lock()
        self._sample_count = 0
        self._count_lock = threading.Lock()
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._pipelines: List[Tuple[str, Any]] = []
        self._pipe_index = 0
        self._pipe_index_lock = threading.Lock()
        self._tokenizer = None

        self._setup_pipelines()

    def _discover_device_dies(self, device: str) -> List[str]:
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup_pipelines(self) -> None:
        from transformers import AutoTokenizer

        target_device = (
            self.config.openvino.device
            if hasattr(self.config, "openvino")
            else None
        )
        if not target_device:
            raise RuntimeError(
                "LlamaMultiDieSUT requires an accelerator device in config"
            )

        # Load HF tokenizer (for encoding generated text -> token IDs)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception:
            logger.debug("Loading tokenizer from meta-llama/Meta-Llama-3.1-8B-Instruct")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Determine die list
        if "," in target_device:
            device_dies = [p.strip() for p in target_device.split(",")]
        elif re.match(r"^.+\.\d+$", target_device):
            device_dies = [target_device]
        else:
            device_dies = self._discover_device_dies(target_device)

        if not device_dies:
            raise RuntimeError(
                f"No dies discovered for device '{target_device}'. "
                f"Check that the device is available."
            )

        # Build device config for LLMPipeline.
        # NOTE: CACHE_DIR is intentionally NOT passed here — LLMPipeline
        # on some accelerators fails at the cache export phase
        # ("Export error … Cannot get program!").  The working reference
        # script does not use model caching either.
        ov_config: Dict[str, Any] = {}
        if hasattr(self.config, "openvino"):
            if hasattr(self.config.openvino, "device_properties"):
                if self.config.openvino.device_properties:
                    for key, value in self.config.openvino.device_properties.items():
                        ov_config[key] = value

        logger.info(f"[Llama] Device config for LLMPipeline: {ov_config}")

        # Generation config -- greedy decoding per MLPerf spec
        self._gen_config = ov_genai.GenerationConfig()
        self._gen_config.max_new_tokens = self.max_new_tokens
        self._gen_config.min_new_tokens = 1

        # SchedulerConfig is required for LLMPipeline on accelerators
        scheduler_config = ov_genai.SchedulerConfig()

        for die in device_dies:
            logger.info(f"[Llama] Creating LLMPipeline for {die} ...")
            pipe = ov_genai.LLMPipeline(
                str(self.model_path), die,
                scheduler_config=scheduler_config,
                **ov_config,
            )
            self._pipelines.append((die, pipe))

        die_names = [name for name, _ in self._pipelines]
        logger.info(
            f"[Llama] {len(self._pipelines)} die(s) ready: {', '.join(die_names)}"
        )

    def _process_sample(
        self, sample_idx: int, pipe: Any,
    ) -> Tuple[str, np.ndarray, int]:
        """Run inference on a single sample using the given pipeline.

        Uses text-based generation: prompt text -> LLMPipeline -> generated text.
        Token IDs are obtained by re-encoding the generated text (needed for
        LoadGen QuerySampleResponse).

        Returns:
            (decoded_text, output_token_ids, n_tokens)
        """
        prompt_text = self.qsl.get_input_text(sample_idx)

        result = pipe.generate(prompt_text, self._gen_config)

        # LLMPipeline.generate(str, ...) returns the generated text
        if isinstance(result, str):
            text = result
        elif hasattr(result, "texts"):
            text = result.texts[0] if result.texts else ""
        else:
            text = str(result)

        # Encode generated text -> int64 token IDs for LoadGen response
        output_tokens = self._tokenizer.encode(text, add_special_tokens=False)
        output_ids = np.array(output_tokens, dtype=np.int64)
        n_tokens = len(output_ids)

        return text, output_ids, n_tokens

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
        self._sample_count = 0

        logger.info(f"[Offline] {total} samples, {num_dies} die(s)")

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        # Round-robin distribution across dies
        from concurrent.futures import ThreadPoolExecutor

        die_batches: List[List[Tuple[Any, int]]] = [[] for _ in range(num_dies)]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        all_results: List[Tuple[Any, int, str, np.ndarray, int]] = []
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]) -> None:
            _, pipe = self._pipelines[die_idx]
            for sample, sample_idx in batch:
                text, output_ids, n_tokens = self._process_sample(sample_idx, pipe)
                with self._predictions_lock:
                    if self._store_predictions:
                        self._predictions[sample_idx] = text
                with self._count_lock:
                    self._sample_count += 1
                with results_lock:
                    all_results.append((sample, sample_idx, text, output_ids, n_tokens))

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = [
                pool.submit(_die_worker, idx, batch)
                for idx, batch in enumerate(die_batches)
                if batch
            ]
            while True:
                done = sum(1 for f in futures if f.done())
                with self._count_lock:
                    count = self._sample_count
                _print_progress(count, total, self._start_time)
                if done == len(futures):
                    break
                time.sleep(0.5)

            for f in futures:
                f.result()  # raise exceptions from workers

        _print_progress(total, total, self._start_time)

        responses = []
        response_arrays = []  # prevent GC before QuerySamplesComplete
        for sample, _idx, _text, output_ids, n_tokens in sorted(all_results, key=lambda r: r[1]):
            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        response_arrays = []  # prevent GC before QuerySamplesComplete

        _, pipe = self._pipelines[0]
        for sample in query_samples:
            sample_idx = sample.index
            text, output_ids, n_tokens = self._process_sample(sample_idx, pipe)

            if self._store_predictions:
                self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)
            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        """Server mode: respond per query with TTFT via FirstTokenComplete."""
        for sample in query_samples:
            sample_idx = sample.index

            with self._pipe_index_lock:
                _name, pipe = self._pipelines[self._pipe_index]
                self._pipe_index = (self._pipe_index + 1) % len(self._pipelines)

            text, output_ids, n_tokens = self._process_sample(sample_idx, pipe)

            if self._store_predictions:
                with self._predictions_lock:
                    self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)

            try:
                lg.FirstTokenComplete([resp])
            except (AttributeError, TypeError):
                pass

            lg.QuerySamplesComplete([resp])
            del resp_arr  # safe to release after QuerySamplesComplete

    def flush_queries(self) -> None:
        pass

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries,
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

    def get_predictions(self) -> Dict[int, str]:
        with self._predictions_lock:
            return self._predictions.copy()

    def set_store_predictions(self, store: bool) -> None:
        self._store_predictions = store

    def reset(self) -> None:
        with self._predictions_lock:
            self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        with self._pipe_index_lock:
            self._pipe_index = 0

    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up each die with a short generation."""
        logger.info(f"[Llama] Warming up {len(self._pipelines)} die(s)...")

        warmup_config = ov_genai.GenerationConfig()
        warmup_config.max_new_tokens = 1
        warmup_config.min_new_tokens = 1

        for die_name, pipe in self._pipelines:
            for _ in range(num_iterations):
                pipe.generate("Hello", warmup_config)
            logger.info(f"  {die_name}: warmed up")
        logger.info("[Llama] Warmup complete")
