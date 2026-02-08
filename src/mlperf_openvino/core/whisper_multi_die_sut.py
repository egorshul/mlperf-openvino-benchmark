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
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    OVModelForSpeechSeq2Seq = None

from .config import BenchmarkConfig, Scenario
from .whisper_sut import _generate_quiet
from ..datasets.librispeech import LibriSpeechQSL

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


class WhisperMultiDieSUT:
    """Whisper multi-die SUT using Optimum-Intel (OVModelForSpeechSeq2Seq).

    Per die: encoder is compiled on the accelerator, decoder is also attempted
    on the accelerator and falls back to CPU if compilation fails.
    Offline mode distributes samples across dies in parallel via round-robin.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for WhisperMultiDieSUT. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.model_path = self.encoder_path.parent
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
        self.processor = None

        self._setup_models()

    def _discover_device_dies(self, device: str) -> List[str]:
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup_models(self) -> None:
        from transformers import AutoProcessor

        target_device = (
            self.config.openvino.device
            if hasattr(self.config, "openvino")
            else None
        )
        if not target_device:
            raise RuntimeError("WhisperMultiDieSUT requires an accelerator device in config")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            logger.debug("Loading processor from openai/whisper-large-v3")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

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

        for die in device_dies:
            logger.info("[Whisper] Loading model for %s ...", die)
            model = self._load_optimum_model()
            self._compile_on_device(model, die)
            self._models.append((die, model))

        die_names = [name for name, _ in self._models]
        logger.info(
            "[Whisper] %d die(s) ready: %s", len(self._models), ", ".join(die_names),
        )

    def _load_optimum_model(self) -> Any:
        model_dir = str(self.model_path)
        try:
            return OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir, device="CPU", ov_config={"CACHE_DIR": ""},
            )
        except Exception as e:
            logger.debug("Default file names failed (%s), trying detected names", e)

        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            device="CPU",
            ov_config={"CACHE_DIR": ""},
            encoder_file_name=self.encoder_path.name,
            decoder_file_name=self.decoder_path.name,
        )

    def _compile_on_device(self, model: Any, die: str) -> None:
        """Compile encoder and decoder on the given die."""
        import openvino as ov

        core = ov.Core()

        ov_config: Dict[str, Any] = {"CACHE_DIR": ""}
        if hasattr(self.config, "openvino") and hasattr(self.config.openvino, "device_properties"):
            for key, value in self.config.openvino.device_properties.items():
                ov_config[key] = value

        encoder_ir = core.read_model(str(self.encoder_path))
        for inp in encoder_ir.inputs:
            if inp.get_partial_shape().is_dynamic:
                encoder_ir.reshape({inp.get_any_name(): [1, 128, 3000]})
                break
        model.encoder.request = core.compile_model(encoder_ir, die, ov_config)
        logger.info("[Whisper %s] Encoder -> %s", die, die)

        try:
            decoder_ir = core.read_model(str(self.decoder_path))
            model.decoder.request = core.compile_model(decoder_ir, die, ov_config)
            logger.info("[Whisper %s] Decoder -> %s", die, die)
        except Exception as exc:
            logger.info(
                "[Whisper %s] Decoder -> CPU (device failed: %s)", die, exc,
            )

    def _process_sample(self, sample_idx: int, model: Any) -> str:
        import torch

        features = self.qsl.get_features(sample_idx)
        input_features = features["input_features"]

        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        generated_ids = _generate_quiet(
            model, input_features,
            max_new_tokens=self.max_new_tokens,
            language="en",
            task="transcribe",
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

        logger.info("[Offline] %d samples, %d die(s)", total, num_dies)

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        # Round-robin distribution across dies
        die_batches: List[List[Tuple[Any, int]]] = [[] for _ in range(num_dies)]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        all_results: List[Tuple[Any, int, str]] = []
        results_lock = threading.Lock()

        def _die_worker(die_idx: int, batch: List[Tuple[Any, int]]):
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
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

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

    def get_predictions(self) -> Dict[int, str]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._model_index = 0
