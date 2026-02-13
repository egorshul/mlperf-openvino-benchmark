"""Llama SUT for single-device (CPU/GPU) inference using OpenVINO GenAI.

Uses openvino_genai.LLMPipeline for OpenVINO-accelerated autoregressive
text generation with KV-cache support. No optimum-intel dependency.

Follows the MLCommons Inference v5.1 reference:
  https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b
"""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

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
        # Older LoadGen versions without n_tokens parameter
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
    return resp, response_array


class LlamaSUT:
    """Llama SUT for single-device inference (CPU/GPU).

    Uses openvino_genai.LLMPipeline for OpenVINO-optimized autoregressive
    generation with KV-cache management.

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
                "openvino-genai is required for LlamaSUT. "
                "Install with: pip install openvino-genai"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        self._store_predictions = True
        self._predictions: Dict[int, str] = {}
        self._sample_count = 0
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._pipeline = None
        self._tokenizer = None

        self._setup_model()

    def _setup_model(self) -> None:
        from transformers import AutoTokenizer

        device = self.config.openvino.device if hasattr(self.config, "openvino") else "CPU"

        ov_config: Dict[str, Any] = {}
        if hasattr(self.config, "openvino"):
            if self.config.openvino.cache_dir:
                ov_config["CACHE_DIR"] = self.config.openvino.cache_dir
            if self.config.openvino.performance_hint:
                ov_config["PERFORMANCE_HINT"] = self.config.openvino.performance_hint
            if self.config.openvino.num_threads > 0:
                ov_config["INFERENCE_NUM_THREADS"] = str(
                    self.config.openvino.num_threads
                )

        logger.info(f"[Llama] Loading model from {self.model_path} on {device}...")

        # Greedy decoding per MLPerf spec
        self._gen_config = ov_genai.GenerationConfig()
        self._gen_config.max_new_tokens = self.max_new_tokens
        self._gen_config.min_new_tokens = 1

        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.max_num_seqs = 1
        scheduler_config.cache_size = 1
        scheduler_config.dynamic_split_fuse = True

        self._pipeline = ov_genai.LLMPipeline(
            str(self.model_path), device,
            scheduler_config=scheduler_config,
            **ov_config,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception:
            logger.debug("Loading tokenizer from meta-llama/Meta-Llama-3.1-8B-Instruct")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"[Llama] Model loaded on {device}")

    def _process_sample(self, sample_idx: int) -> tuple:
        """Run inference on a single sample.

        Uses text-based generation: prompt text -> LLMPipeline -> generated text.
        Token IDs are obtained by re-encoding the generated text (needed for
        LoadGen QuerySampleResponse).

        Returns:
            (decoded_text, output_token_ids, n_tokens)
        """
        prompt_text = self.qsl.get_input_text(sample_idx)

        result = self._pipeline.generate(prompt_text, self._gen_config)

        if isinstance(result, str):
            text = result
        elif hasattr(result, "texts"):
            text = result.texts[0] if result.texts else ""
        else:
            text = str(result)

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
        self._start_time = time.time()
        self._sample_count = 0

        device = self.config.openvino.device if hasattr(self.config, "openvino") else "CPU"
        logger.info(f"[Offline] Processing {total} samples on {device}")

        responses = []
        response_arrays = []  # prevent GC before QuerySamplesComplete

        for sample in query_samples:
            sample_idx = sample.index
            text, output_ids, n_tokens = self._process_sample(sample_idx)

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
        """Server mode: respond per query, report FirstTokenComplete for TTFT."""
        for sample in query_samples:
            sample_idx = sample.index
            text, output_ids, n_tokens = self._process_sample(sample_idx)

            if self._store_predictions:
                self._predictions[sample_idx] = text

            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)

            try:
                lg.FirstTokenComplete([resp])
            except (AttributeError, TypeError):
                pass  # Older LoadGen without FirstTokenComplete

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
        return self._predictions.copy()

    def set_store_predictions(self, store: bool) -> None:
        self._store_predictions = store

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def warmup(self, num_iterations: int = 5) -> None:
        """Warm up the model with a short generation."""
        logger.info(f"[Llama] Warming up ({num_iterations} iterations)...")

        warmup_config = ov_genai.GenerationConfig()
        warmup_config.max_new_tokens = 1
        warmup_config.min_new_tokens = 1

        for _ in range(num_iterations):
            self._pipeline.generate("Hello", warmup_config)
        logger.info("[Llama] Warmup complete")
