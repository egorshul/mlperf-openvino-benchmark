"""Llama SUT for single-device (CPU/GPU) inference using Optimum-Intel.

Uses OVModelForCausalLM from optimum-intel for OpenVINO-accelerated
autoregressive text generation with KV-cache support.
"""

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
    from optimum.intel.openvino import OVModelForCausalLM

    OPTIMUM_CAUSAL_LM_AVAILABLE = True
except ImportError:
    OPTIMUM_CAUSAL_LM_AVAILABLE = False
    OVModelForCausalLM = None

from .config import BenchmarkConfig, Scenario
from ..datasets.open_orca import OpenOrcaQSL

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


class LlamaSUT:
    """Llama SUT for single-device inference (CPU/GPU).

    Uses Optimum-Intel OVModelForCausalLM for OpenVINO-optimized
    autoregressive generation with stateful KV-cache.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: OpenOrcaQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 1024,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")
        if not OPTIMUM_CAUSAL_LM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for LlamaSUT. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        self._predictions: Dict[int, str] = {}
        self._sample_count = 0
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._model = None
        self._tokenizer = None

        self._setup_model()

    def _setup_model(self) -> None:
        from transformers import AutoTokenizer

        device = self.config.openvino.device if hasattr(self.config, "openvino") else "CPU"

        ov_config: Dict[str, Any] = {"CACHE_DIR": ""}
        if hasattr(self.config, "openvino"):
            if self.config.openvino.performance_hint:
                ov_config["PERFORMANCE_HINT"] = self.config.openvino.performance_hint
            if self.config.openvino.num_threads > 0:
                ov_config["INFERENCE_NUM_THREADS"] = str(
                    self.config.openvino.num_threads
                )

        logger.info(f"[Llama] Loading model from {self.model_path} on {device}...")

        self._model = OVModelForCausalLM.from_pretrained(
            str(self.model_path),
            device=device,
            ov_config=ov_config,
            compile=True,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception:
            logger.debug("Loading tokenizer from meta-llama/Llama-3.1-8B-Instruct")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct"
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"[Llama] Model loaded on {device}")

    def _process_sample(self, sample_idx: int) -> str:
        """Run inference on a single sample: tokenize → generate → decode."""
        import torch

        features = self.qsl.get_features(sample_idx)
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).long()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_len = input_ids.shape[-1]

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding per MLPerf spec
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens (exclude input prompt)
        new_tokens = generated_ids[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

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

        logger.info(f"[Offline] Processing {total} samples sequentially on CPU")

        responses = []
        arrays = []

        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            text_bytes = text.encode("utf-8")
            arr = array.array("B", text_bytes)
            arrays.append(arr)
            bi = arr.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            text_bytes = text.encode("utf-8")
            arr = array.array("B", text_bytes)
            bi = arr.buffer_info()
            lg.QuerySamplesComplete(
                [lg.QuerySampleResponse(sample.id, bi[0], bi[1])]
            )

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
        pass  # Always store for accuracy mode

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0

    def warmup(self, num_iterations: int = 5) -> None:
        """Warm up the model with dummy inputs."""
        import torch

        logger.info(f"[Llama] Warming up ({num_iterations} iterations)...")
        dummy_input = torch.ones(1, 32, dtype=torch.long)
        dummy_mask = torch.ones(1, 32, dtype=torch.long)

        for _ in range(num_iterations):
            with torch.no_grad():
                self._model.generate(
                    input_ids=dummy_input,
                    attention_mask=dummy_mask,
                    max_new_tokens=1,
                    do_sample=False,
                )
        logger.info("[Llama] Warmup complete")
