"""Llama multi-die SUT for accelerator inference (NPU/VPU/XPU).

Distributes text generation across multiple accelerator dies using
OVModelForCausalLM from Optimum-Intel. Each die gets a compiled copy
of the model; samples are dispatched round-robin across dies.

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
    from optimum.intel.openvino import OVModelForCausalLM

    OPTIMUM_CAUSAL_LM_AVAILABLE = True
except ImportError:
    OPTIMUM_CAUSAL_LM_AVAILABLE = False
    OVModelForCausalLM = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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


def _make_response(sample_id: int, text: str, n_tokens: int):
    """Build a QuerySampleResponse with n_tokens (required for LLM benchmarks).

    Returns:
        (response, response_array) — caller must keep response_array alive
        until QuerySamplesComplete processes the response buffer.
    """
    text_bytes = text.encode("utf-8")
    response_array = array.array("B", text_bytes)
    bi = response_array.buffer_info()
    try:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1], n_tokens)
    except TypeError:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
    return resp, response_array


class LlamaMultiDieSUT:
    """Llama multi-die SUT using Optimum-Intel (OVModelForCausalLM).

    Per die: model is loaded and compiled on the accelerator.
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
        if not OPTIMUM_CAUSAL_LM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for LlamaMultiDieSUT. "
                "Install with: pip install optimum[openvino]"
            )
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LlamaMultiDieSUT")

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

        self._models: List[Tuple[str, Any]] = []
        self._model_index = 0
        self._model_index_lock = threading.Lock()
        self._tokenizer = None

        self._setup_models()

    def _discover_device_dies(self, device: str) -> List[str]:
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup_models(self) -> None:
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

        # Load tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception:
            logger.debug("Loading tokenizer from meta-llama/Llama-3.1-8B-Instruct")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct"
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

        ov_config: Dict[str, Any] = {}
        if hasattr(self.config, "openvino"):
            if self.config.openvino.cache_dir:
                ov_config["CACHE_DIR"] = self.config.openvino.cache_dir
            if hasattr(self.config.openvino, "device_properties"):
                if self.config.openvino.device_properties:
                    for key, value in self.config.openvino.device_properties.items():
                        ov_config[key] = value

        for die in device_dies:
            logger.info(f"[Llama] Loading model for {die} ...")
            model = OVModelForCausalLM.from_pretrained(
                str(self.model_path),
                device=die,
                ov_config=ov_config,
                compile=True,
            )
            self._models.append((die, model))

        die_names = [name for name, _ in self._models]
        logger.info(
            f"[Llama] {len(self._models)} die(s) ready: {', '.join(die_names)}"
        )

    def _process_sample(self, sample_idx: int, model: Any) -> Tuple[str, int]:
        """Run inference on a single sample using the given model.

        Returns:
            (decoded_text, n_tokens) — generated text and token count.
        """
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
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,  # Greedy decoding per MLPerf spec
                num_beams=1,
            )

        # Extract only the newly generated tokens (exclude input prompt)
        new_token_ids = generated_ids[0, input_len:]
        n_tokens = len(new_token_ids)
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return text, n_tokens

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        num_dies = len(self._models)
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

        all_results: List[Tuple[Any, int, str, int]] = []  # (sample, idx, text, n_tokens)
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]) -> None:
            _, model = self._models[die_idx]
            for sample, sample_idx in batch:
                text, n_tokens = self._process_sample(sample_idx, model)
                with self._predictions_lock:
                    if self._store_predictions:
                        self._predictions[sample_idx] = text
                with self._count_lock:
                    self._sample_count += 1
                with results_lock:
                    all_results.append((sample, sample_idx, text, n_tokens))

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
        for sample, _idx, text, n_tokens in sorted(all_results, key=lambda r: r[1]):
            resp, resp_arr = _make_response(sample.id, text, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        response_arrays = []  # prevent GC before QuerySamplesComplete

        _, model = self._models[0]
        for sample in query_samples:
            sample_idx = sample.index
            text, n_tokens = self._process_sample(sample_idx, model)

            if self._store_predictions:
                self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, text, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)
            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        """Server mode: respond per query with TTFT via FirstTokenComplete."""
        for sample in query_samples:
            sample_idx = sample.index

            with self._model_index_lock:
                _name, model = self._models[self._model_index]
                self._model_index = (self._model_index + 1) % len(self._models)

            text, n_tokens = self._process_sample(sample_idx, model)

            if self._store_predictions:
                with self._predictions_lock:
                    self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, text, n_tokens)

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
        with self._model_index_lock:
            self._model_index = 0

    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up each die with dummy inputs."""
        logger.info(f"[Llama] Warming up {len(self._models)} die(s)...")
        dummy_input = torch.ones(1, 32, dtype=torch.long)
        dummy_mask = torch.ones(1, 32, dtype=torch.long)

        for die_name, model in self._models:
            for _ in range(num_iterations):
                with torch.no_grad():
                    model.generate(
                        input_ids=dummy_input,
                        attention_mask=dummy_mask,
                        max_new_tokens=1,
                        do_sample=False,
                    )
            logger.info(f"  {die_name}: warmed up")
        logger.info("[Llama] Warmup complete")
