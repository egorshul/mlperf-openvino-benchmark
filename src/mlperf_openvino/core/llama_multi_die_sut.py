"""Llama multi-die SUT for accelerator inference (NPU/VPU/XPU).

Distributes text generation across multiple accelerator dies using
OVModelForCausalLM from Optimum-Intel. Each die gets a compiled copy
of the model; samples are dispatched round-robin across dies.

Follows the same pattern as WhisperMultiDieSUT.
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


class LlamaMultiDieSUT:
    """Llama multi-die SUT using Optimum-Intel (OVModelForCausalLM).

    Per die: model is loaded and compiled on the accelerator.
    Offline mode distributes samples across dies in parallel via round-robin.
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
                "optimum-intel is required for LlamaMultiDieSUT. "
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

        self._models: List[Tuple[str, Any]] = []
        self._model_index = 0
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

        ov_config: Dict[str, Any] = {"CACHE_DIR": ""}
        if hasattr(self.config, "openvino") and hasattr(
            self.config.openvino, "device_properties"
        ):
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

    def _process_sample(self, sample_idx: int, model: Any) -> str:
        """Run inference on a single sample using the given model."""
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
            generated_ids = model.generate(
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

        all_results: List[Tuple[Any, int, str]] = []
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]) -> None:
            _, model = self._models[die_idx]
            for sample, sample_idx in batch:
                text = self._process_sample(sample_idx, model)
                self._predictions[sample_idx] = text
                self._sample_count += 1
                with results_lock:
                    all_results.append((sample, sample_idx, text))

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = [
                pool.submit(_die_worker, idx, batch)
                for idx, batch in enumerate(die_batches)
                if batch
            ]
            while True:
                done = sum(1 for f in futures if f.done())
                _print_progress(self._sample_count, total, self._start_time)
                if done == len(futures):
                    break
                time.sleep(0.5)

            for f in futures:
                f.result()

        _print_progress(total, total, self._start_time)

        responses = []
        arrays = []
        for sample, _idx, text in sorted(all_results, key=lambda r: r[1]):
            text_bytes = text.encode("utf-8")
            arr = array.array("B", text_bytes)
            arrays.append(arr)
            bi = arr.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        arrays = []

        _, model = self._models[0]
        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx, model)
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
            _name, model = self._models[self._model_index]
            self._model_index = (self._model_index + 1) % len(self._models)

            text = self._process_sample(sample_idx, model)
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
        self._model_index = 0
