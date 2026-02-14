import array
import gc
import logging
import multiprocessing
import queue
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


def _die_worker_fn(
    die_name: str,
    model_path: str,
    ov_config: Dict[str, Any],
    max_new_tokens: int,
    tokenizer_path: str,
    input_queue,
    output_queue,
    ready_event,
) -> None:
    import openvino_genai as _ov_genai
    from transformers import AutoTokenizer

    scheduler_config = _ov_genai.SchedulerConfig()
    scheduler_config.max_num_seqs = 1
    scheduler_config.cache_size = 1
    scheduler_config.dynamic_split_fuse = True

    pipe = _ov_genai.LLMPipeline(
        model_path, die_name,
        scheduler_config=scheduler_config,
        **ov_config,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = _ov_genai.GenerationConfig()
    gen_config.max_new_tokens = max_new_tokens
    gen_config.min_new_tokens = 1

    warmup_cfg = _ov_genai.GenerationConfig()
    warmup_cfg.max_new_tokens = 1
    warmup_cfg.min_new_tokens = 1
    for _ in range(3):
        pipe.generate("Hello", warmup_cfg)

    ready_event.set()

    while True:
        item = input_queue.get()
        if item is None:
            break

        item_idx, sample_id, prompt_text = item
        result = pipe.generate(prompt_text, gen_config)

        output_ids = None
        if hasattr(result, "m_generation_ids") and result.m_generation_ids:
            output_ids = np.array(result.m_generation_ids, dtype=np.int32)

        if output_ids is not None:
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
        elif isinstance(result, str):
            text = result
        elif hasattr(result, "texts") and result.texts:
            text = result.texts[0]
        else:
            text = str(result)

        if output_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            output_ids = np.array(tokens, dtype=np.int32)

        n_tokens = len(output_ids)
        output_queue.put((
            item_idx, sample_id, text,
            output_ids.tobytes(), n_tokens,
        ))


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
    token_bytes = token_ids.astype(np.int32).tobytes()
    response_array = array.array("B", token_bytes)
    bi = response_array.buffer_info()
    try:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1], n_tokens)
    except TypeError:
        resp = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
    return resp, response_array


class LlamaMultiDieSUT:

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
        self._sample_count = 0
        self._query_count = 0
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._workers: List[Tuple[str, multiprocessing.Process, Any, Any]] = []
        self._pipelines: List[Tuple[str, Any]] = []
        self._pipe_index = 0
        self._pipe_index_lock = threading.Lock()

        self._tokenizer = None

        self._setup()

    @staticmethod
    def _get_available_memory_gb() -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass
        return -1.0

    def _discover_device_dies(self, device: str) -> List[str]:
        import openvino as ov
        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup(self) -> None:
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
            if hasattr(self.config.openvino, "device_properties"):
                if self.config.openvino.device_properties:
                    for k, v in self.config.openvino.device_properties.items():
                        ov_config[k] = v

        if self.scenario == Scenario.OFFLINE and len(device_dies) > 1:
            self._setup_workers(device_dies, ov_config)
        else:
            self._setup_pipelines_inprocess(device_dies, ov_config)

    def _setup_workers(
        self, device_dies: List[str], ov_config: Dict[str, Any],
    ) -> None:
        ctx = multiprocessing.get_context("spawn")

        logger.info(
            f"[Llama] Spawning {len(device_dies)} worker process(es)..."
        )

        pending: List[Tuple[str, multiprocessing.Process, Any, Any, Any]] = []

        for die_name in device_dies:
            input_q = ctx.Queue()
            output_q = ctx.Queue()
            ready = ctx.Event()

            p = ctx.Process(
                target=_die_worker_fn,
                args=(
                    die_name, str(self.model_path), ov_config,
                    self.max_new_tokens, str(self.model_path),
                    input_q, output_q, ready,
                ),
                daemon=True,
            )
            p.start()
            pending.append((die_name, p, input_q, output_q, ready))
            logger.info(f"  {die_name}: spawned (pid={p.pid})")

        for die_name, p, input_q, output_q, ready in pending:
            if not ready.wait(timeout=600):
                raise RuntimeError(
                    f"Worker for {die_name} did not become ready in 600 s"
                )
            self._workers.append((die_name, p, input_q, output_q))
            logger.info(f"  {die_name}: model loaded & warmed up")

        logger.info(
            f"[Llama] {len(self._workers)} worker(s) ready"
        )

    def _setup_pipelines_inprocess(
        self, device_dies: List[str], ov_config: Dict[str, Any],
    ) -> None:
        self._gen_config = ov_genai.GenerationConfig()
        self._gen_config.max_new_tokens = self.max_new_tokens
        self._gen_config.min_new_tokens = 1

        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.max_num_seqs = 1
        scheduler_config.cache_size = 1
        scheduler_config.dynamic_split_fuse = True

        for i, die in enumerate(device_dies):
            if i > 0:
                gc.collect()
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except Exception:
                    pass

            avail_gb = self._get_available_memory_gb()
            logger.info(
                f"[Llama] Creating LLMPipeline for {die} "
                f"(available RAM: {avail_gb:.1f} GB) ..."
            )
            pipe = ov_genai.LLMPipeline(
                str(self.model_path), die,
                scheduler_config=scheduler_config,
                **ov_config,
            )
            self._pipelines.append((die, pipe))

        die_names = [n for n, _ in self._pipelines]
        logger.info(
            f"[Llama] {len(self._pipelines)} pipeline(s) ready: "
            f"{', '.join(die_names)}"
        )

    def _parse_generation_result(
        self, gen_result: Any,
    ) -> Tuple[str, np.ndarray, int]:
        output_ids = None
        if hasattr(gen_result, "m_generation_ids") and gen_result.m_generation_ids:
            output_ids = np.array(gen_result.m_generation_ids, dtype=np.int32)

        if output_ids is not None:
            text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        elif isinstance(gen_result, str):
            text = gen_result
        elif hasattr(gen_result, "texts") and gen_result.texts:
            text = gen_result.texts[0]
        else:
            text = str(gen_result)

        if output_ids is None:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            output_ids = np.array(tokens, dtype=np.int32)

        return text, output_ids, len(output_ids)

    def _process_sample(
        self, sample_idx: int, pipe: Any,
    ) -> Tuple[str, np.ndarray, int]:
        prompt_text = self.qsl.get_input_text(sample_idx)
        result = pipe.generate(prompt_text, self._gen_config)
        return self._parse_generation_result(result)

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

        if self._workers:
            logger.info(
                f"[Offline] {total} samples, {len(self._workers)} worker process(es)"
            )
            self._issue_offline_multiprocess(query_samples)
        else:
            logger.info(
                f"[Offline] {total} samples, {len(self._pipelines)} die(s) (in-process)"
            )
            self._issue_offline_sequential(query_samples)

    def _issue_offline_multiprocess(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        num_workers = len(self._workers)

        for i, sample in enumerate(query_samples):
            worker_idx = i % num_workers
            prompt = self.qsl.get_input_text(sample.index)
            _, _, input_q, _ = self._workers[worker_idx]
            input_q.put((i, sample.id, prompt))

        collected: Dict[int, Tuple] = {}

        while len(collected) < total:
            for die_name, p, _, _ in self._workers:
                if not p.is_alive():
                    raise RuntimeError(
                        f"Worker {die_name} died (exit code {p.exitcode})"
                    )

            for _, _, _, output_q in self._workers:
                try:
                    while True:
                        result = output_q.get_nowait()
                        collected[result[0]] = result
                except queue.Empty:
                    pass

            self._sample_count = len(collected)
            _print_progress(self._sample_count, total, self._start_time)

            if len(collected) < total:
                time.sleep(0.1)

        _print_progress(total, total, self._start_time)

        responses = []
        response_arrays = []
        for i, sample in enumerate(query_samples):
            _, _, text, token_bytes, n_tokens = collected[i]
            output_ids = np.frombuffer(token_bytes, dtype=np.int32).copy()

            if self._store_predictions:
                self._predictions[sample.index] = text

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)

        lg.QuerySamplesComplete(responses)

    def _issue_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        response_arrays = []

        _, pipe = self._pipelines[0]
        for sample in query_samples:
            text, output_ids, n_tokens = self._process_sample(
                sample.index, pipe
            )

            if self._store_predictions:
                self._predictions[sample.index] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)
            responses.append(resp)
            response_arrays.append(resp_arr)
            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index

            with self._pipe_index_lock:
                _name, pipe = self._pipelines[self._pipe_index]
                self._pipe_index = (
                    (self._pipe_index + 1) % len(self._pipelines)
                )

            text, output_ids, n_tokens = self._process_sample(
                sample_idx, pipe
            )

            if self._store_predictions:
                self._predictions[sample_idx] = text
            self._sample_count += 1

            resp, resp_arr = _make_response(sample.id, output_ids, n_tokens)

            try:
                lg.FirstTokenComplete([resp])
            except (AttributeError, TypeError):
                pass

            lg.QuerySamplesComplete([resp])
            del resp_arr

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
        with self._pipe_index_lock:
            self._pipe_index = 0

    def warmup(self, num_iterations: int = 3) -> None:
        if self._workers:
            logger.info(
                f"[Llama] {len(self._workers)} worker(s) already "
                f"warmed up during init"
            )
            return

        logger.info(
            f"[Llama] Warming up {len(self._pipelines)} pipeline(s)..."
        )
        warmup_config = ov_genai.GenerationConfig()
        warmup_config.max_new_tokens = 1
        warmup_config.min_new_tokens = 1
        for die_name, pipe in self._pipelines:
            for _ in range(num_iterations):
                pipe.generate("Hello", warmup_config)
            logger.info(f"  {die_name}: warmed up")
        logger.info("[Llama] Warmup complete")

    def shutdown(self) -> None:
        for _, p, input_q, _ in self._workers:
            try:
                input_q.put_nowait(None)
            except Exception:
                pass
        for _, p, _, _ in self._workers:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
