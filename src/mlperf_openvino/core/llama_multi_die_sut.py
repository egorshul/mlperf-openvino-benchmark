"""Llama multi-die SUT for accelerator inference.

Uses the OpenVINO Core API directly so that it works with **any**
accelerator device name (not just ``"NPU"``).  The model IR is read
once, reshaped to fully-static dimensions, and compiled separately
for each die.  Text generation is a manual greedy loop over the
stateful (KV-cache) model.

Static shapes used (the only "dynamic" axis — attention-mask
sequence length — is pre-allocated to the maximum context length):

    input_ids      : [1, 1]
    position_ids   : [1, 1]
    attention_mask  : [1, max_prompt_len + max_new_tokens]
    beam_idx       : [1]

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import openvino as ov

    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False
    ov = None

try:
    import mlperf_loadgen as lg

    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, Scenario
from ..datasets.cnn_dailymail import CnnDailyMailQSL

logger = logging.getLogger(__name__)

# OpenVINO element-type → numpy dtype
_OV_TO_NP = {}
if OV_AVAILABLE:
    _OV_TO_NP = {
        ov.Type.i32: np.int32,
        ov.Type.i64: np.int64,
        ov.Type.f32: np.float32,
        ov.Type.f16: np.float16,
    }


# ── helpers ────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _print_progress(completed: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    if completed >= total:
        line = (
            f"\r[Inference] {completed}/{total} | "
            f"{rate:.2f} samples/s | {_fmt_time(elapsed)}"
        )
        print(line, file=sys.stderr, flush=True)
        print(file=sys.stderr)
    else:
        eta = (total - completed) / rate if rate > 0 else 0.0
        line = (
            f"\r[Inference] {completed}/{total} | "
            f"{rate:.2f} samples/s | ETA {_fmt_time(eta)}"
        )
        print(line, end="", file=sys.stderr, flush=True)


def _make_response(sample_id: int, token_ids: np.ndarray, n_tokens: int):
    """Build a QuerySampleResponse with int64 token IDs."""
    token_bytes = token_ids.astype(np.int64).tobytes()
    response_array = array.array("B", token_bytes)
    bi = response_array.buffer_info()
    try:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1], n_tokens)
    except TypeError:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
    return resp, response_array


def _find_model_xml(model_dir: Path) -> Path:
    """Locate the OpenVINO IR ``.xml`` file inside *model_dir*."""
    for name in ("openvino_model.xml", "model.xml"):
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No OpenVINO IR found in {model_dir}. "
        "Expected openvino_model.xml or model.xml."
    )


def _np_dtype(ov_type) -> np.dtype:
    """Map an ``ov.Type`` to a numpy dtype (fallback: int64)."""
    return _OV_TO_NP.get(ov_type, np.int64)


# ── SUT ────────────────────────────────────────────────────────────────

class LlamaMultiDieSUT:
    """Llama multi-die SUT — OpenVINO Core API with manual greedy generation.

    Works with any accelerator device name.  Each die gets its own
    ``CompiledModel`` / ``InferRequest``; samples are dispatched
    round-robin across dies in Offline mode.

    Per MLPerf v5.1 Llama 3.1 8B spec:
      - Task: text summarization (CNN-DailyMail)
      - max_new_tokens: 128
      - Greedy decoding
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
        if not OV_AVAILABLE:
            raise ImportError("openvino is required for LlamaMultiDieSUT")
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

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

        # Per-die inference contexts: (die_name, infer_request)
        self._dies: List[Tuple[str, Any]] = []
        self._die_index = 0
        self._die_index_lock = threading.Lock()

        self._tokenizer = None
        self._eos_token_id: Optional[int] = None
        self._max_context_len = 0
        # Maps logical name → actual IR input node name
        self._input_map: Dict[str, str] = {}
        # Maps logical name → numpy dtype for that input
        self._input_dtypes: Dict[str, np.dtype] = {}

        self._setup()

    # ── discovery ──────────────────────────────────────────────────

    @staticmethod
    def _discover_device_dies(device: str) -> List[str]:
        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    # ── setup ──────────────────────────────────────────────────────

    def _setup(self) -> None:
        from transformers import AutoTokenizer

        # ── tokenizer ──────────────────────────────────────────────
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path)
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._eos_token_id = self._tokenizer.eos_token_id

        # ── read & reshape model ───────────────────────────────────
        core = ov.Core()
        model_xml = _find_model_xml(self.model_path)
        logger.info(f"[Llama] Reading model from {model_xml}")
        model = core.read_model(str(model_xml))

        max_prompt_len = self.qsl.get_max_input_length()
        self._max_context_len = max_prompt_len + self.max_new_tokens
        logger.info(
            f"[Llama] max_prompt_len={max_prompt_len}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"max_context_len={self._max_context_len}"
        )

        has_beam_idx = False
        has_past_kv = False

        new_shapes: Dict[str, list] = {}
        for inp in model.inputs:
            names = set(inp.get_names())
            name = inp.get_any_name()
            dtype = _np_dtype(inp.get_element_type())

            if any("input_ids" in n for n in names):
                self._input_map["input_ids"] = name
                self._input_dtypes["input_ids"] = dtype
                new_shapes[name] = [1, 1]
            elif any("attention_mask" in n for n in names):
                self._input_map["attention_mask"] = name
                self._input_dtypes["attention_mask"] = dtype
                new_shapes[name] = [1, self._max_context_len]
            elif any("position_ids" in n for n in names):
                self._input_map["position_ids"] = name
                self._input_dtypes["position_ids"] = dtype
                new_shapes[name] = [1, 1]
            elif any("beam_idx" in n for n in names):
                has_beam_idx = True
                self._input_map["beam_idx"] = name
                self._input_dtypes["beam_idx"] = dtype
                new_shapes[name] = [1]
            elif any("past_key_values" in n for n in names):
                has_past_kv = True
                # collapse dynamic dims to 1
                shape = inp.get_partial_shape()
                static = []
                for dim in shape:
                    static.append(dim.get_length() if dim.is_static else 1)
                new_shapes[name] = static
            else:
                # Unknown input — make every dim static (keep static dims,
                # collapse dynamic dims to 1).
                shape = inp.get_partial_shape()
                static = []
                for dim in shape:
                    static.append(dim.get_length() if dim.is_static else 1)
                new_shapes[name] = static

        if has_past_kv and not has_beam_idx:
            raise ValueError(
                "Non-stateful with-past model detected (explicit past_key_values "
                "inputs).  The multi-die SUT requires a stateful model whose "
                "KV-cache is managed via OpenVINO state variables.  "
                "Re-export with --stateful, or use the single-device LlamaSUT "
                "(CPU/GPU) which handles non-stateful models via OVModelForCausalLM."
            )

        logger.info(f"[Llama] Reshaping model to static shapes: {new_shapes}")
        model.reshape(new_shapes)

        # ── die discovery ──────────────────────────────────────────
        target_device = self.config.openvino.device
        if "," in target_device:
            dies = [p.strip() for p in target_device.split(",")]
        elif re.match(r"^.+\.\d+$", target_device):
            dies = [target_device]
        else:
            dies = self._discover_device_dies(target_device)
            if not dies:
                dies = [target_device]

        # ── compile per die ────────────────────────────────────────
        compile_props: Dict[str, Any] = {}
        if self.config.openvino.cache_dir:
            compile_props["CACHE_DIR"] = self.config.openvino.cache_dir
        if hasattr(self.config.openvino, "device_properties"):
            if self.config.openvino.device_properties:
                compile_props.update(self.config.openvino.device_properties)

        for die in dies:
            logger.info(f"[Llama] Compiling for {die} ...")
            compiled = core.compile_model(model, die, compile_props)
            req = compiled.create_infer_request()
            self._dies.append((die, req))

        die_names = [n for n, _ in self._dies]
        logger.info(
            f"[Llama] {len(self._dies)} die(s) ready: {', '.join(die_names)}"
        )

    # ── greedy generation ──────────────────────────────────────────

    def _generate_greedy(
        self,
        token_ids: List[int],
        infer_req: Any,
    ) -> List[int]:
        """Token-by-token greedy generation on a stateful OV model.

        Args:
            token_ids: prompt token IDs (flat list).
            infer_req: an ``ov.InferRequest`` with KV-cache states.

        Returns:
            List of **newly generated** token IDs (excluding the prompt).
        """
        infer_req.reset_state()

        prompt_len = len(token_ids)
        ids_dtype = self._input_dtypes.get("input_ids", np.int64)
        mask_dtype = self._input_dtypes.get("attention_mask", np.int64)
        pos_dtype = self._input_dtypes.get("position_ids", np.int64)
        beam_dtype = self._input_dtypes.get("beam_idx", np.int32)

        attention_mask = np.zeros(
            (1, self._max_context_len), dtype=mask_dtype,
        )

        def _infer_token(token: int, pos: int) -> np.ndarray:
            attention_mask[0, pos] = 1
            inputs = {
                self._input_map["input_ids"]: np.array(
                    [[token]], dtype=ids_dtype,
                ),
                self._input_map["attention_mask"]: attention_mask,
                self._input_map["position_ids"]: np.array(
                    [[pos]], dtype=pos_dtype,
                ),
            }
            if "beam_idx" in self._input_map:
                inputs[self._input_map["beam_idx"]] = np.array(
                    [0], dtype=beam_dtype,
                )
            infer_req.infer(inputs)
            return infer_req.get_output_tensor(0).data  # logits

        # ── prefill (one token at a time) ──────────────────────────
        for pos in range(prompt_len):
            logits = _infer_token(token_ids[pos], pos)

        # First generated token
        next_token = int(np.argmax(logits[0, 0, :]))
        if next_token == self._eos_token_id:
            return []

        generated: List[int] = [next_token]

        # ── decode ─────────────────────────────────────────────────
        for step in range(1, self.max_new_tokens):
            pos = prompt_len + step - 1
            logits = _infer_token(next_token, pos)

            next_token = int(np.argmax(logits[0, 0, :]))
            if next_token == self._eos_token_id:
                break
            generated.append(next_token)

        return generated

    # ── sample processing ──────────────────────────────────────────

    def _process_sample(
        self,
        sample_idx: int,
        infer_req: Any,
    ) -> Tuple[str, np.ndarray, int]:
        features = self.qsl.get_features(sample_idx)
        input_ids = features["input_ids"]
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.reshape(-1).tolist()

        gen_ids = self._generate_greedy(input_ids, infer_req)

        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        output_ids = np.array(gen_ids, dtype=np.int64)
        n_tokens = len(gen_ids)
        return text, output_ids, n_tokens

    # ── LoadGen callbacks ──────────────────────────────────────────

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        num_dies = len(self._dies)
        self._start_time = time.time()
        self._sample_count = 0

        logger.info(f"[Offline] {total} samples, {num_dies} die(s)")

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        from concurrent.futures import ThreadPoolExecutor

        die_batches: List[List[Tuple[Any, int]]] = [
            [] for _ in range(num_dies)
        ]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        all_results: List[Tuple[Any, int, str, np.ndarray, int]] = []
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]) -> None:
            _, req = self._dies[die_idx]
            for sample, sample_idx in batch:
                text, output_ids, n_tokens = self._process_sample(
                    sample_idx, req,
                )
                with self._predictions_lock:
                    if self._store_predictions:
                        self._predictions[sample_idx] = text
                with self._count_lock:
                    self._sample_count += 1
                with results_lock:
                    all_results.append(
                        (sample, sample_idx, text, output_ids, n_tokens)
                    )

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
                f.result()

        _print_progress(total, total, self._start_time)

        responses = []
        response_arrays = []
        for sample, _idx, _text, output_ids, n_tokens in sorted(
            all_results, key=lambda r: r[1],
        ):
            resp, arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(arr)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(
        self, query_samples: List[Any],
    ) -> None:
        total = len(query_samples)
        responses = []
        response_arrays = []

        _, req = self._dies[0]
        for sample in query_samples:
            sample_idx = sample.index
            text, output_ids, n_tokens = self._process_sample(
                sample_idx, req,
            )
            if self._store_predictions:
                self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(arr)
            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index

            with self._die_index_lock:
                _name, req = self._dies[self._die_index]
                self._die_index = (self._die_index + 1) % len(self._dies)

            text, output_ids, n_tokens = self._process_sample(
                sample_idx, req,
            )
            if self._store_predictions:
                with self._predictions_lock:
                    self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, arr = _make_response(sample.id, output_ids, n_tokens)
            try:
                lg.FirstTokenComplete([resp])
            except (AttributeError, TypeError):
                pass
            lg.QuerySamplesComplete([resp])
            del arr

    def flush_queries(self) -> None:
        pass

    # ── LoadGen handles ────────────────────────────────────────────

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

    # ── state ──────────────────────────────────────────────────────

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
        with self._die_index_lock:
            self._die_index = 0

    def warmup(self, num_iterations: int = 2) -> None:
        """Warm up each die with a short generation."""
        logger.info(f"[Llama] Warming up {len(self._dies)} die(s)...")
        # Minimal prompt: one token
        dummy_ids = [self._tokenizer.bos_token_id or 1]
        for die_name, req in self._dies:
            for _ in range(num_iterations):
                self._generate_greedy(dummy_ids, req)
            logger.info(f"  {die_name}: warmed up")
        logger.info("[Llama] Warmup complete")
