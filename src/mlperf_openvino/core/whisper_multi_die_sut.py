"""Whisper ASR Multi-Die System Under Test."""

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


class WhisperMultiDieSUT:
    """Whisper ASR SUT for multi-die accelerators.

    Loads one OVModelForSpeechSeq2Seq per die with the encoder compiled
    on the accelerator and the decoder on CPU. Offline mode distributes
    samples across dies in parallel; Server mode uses round-robin dispatch.
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

    # -- Device discovery & model loading -----------------------------------

    def _discover_device_dies(self, device: str) -> List[str]:
        """Return sorted list of sub-device identifiers (e.g. NPU.0, NPU.1)."""
        import openvino as ov

        core = ov.Core()
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        return sorted(d for d in core.available_devices if pattern.match(d))

    def _setup_models(self) -> None:
        """Load one model per accelerator die."""
        from transformers import AutoProcessor

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for WhisperMultiDieSUT. "
                "Install with: pip install optimum[openvino]"
            )

        target_device = (
            self.config.openvino.device
            if hasattr(self.config, "openvino")
            else "CPU"
        )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            logger.info("Loading processor from openai/whisper-large-v3")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        if target_device == "CPU":
            device_dies = ["CPU"]
        elif re.match(r"^.+\.\d+$", target_device):
            device_dies = [target_device]
        else:
            device_dies = self._discover_device_dies(target_device)
            if not device_dies:
                logger.warning("No %s dies found, falling back to CPU", target_device)
                device_dies = ["CPU"]

        for die in device_dies:
            try:
                logger.info("Loading Whisper model for %s ...", die)
                model = self._load_optimum_model()
                if die != "CPU":
                    self._patch_encoder_device(model, die)
                self._models.append((die, model))
                logger.info("Model ready on %s", die)
            except Exception:
                logger.exception("Failed to create model for %s", die)

        if not self._models:
            logger.warning("No accelerator models loaded, falling back to CPU")
            model = self._load_optimum_model()
            self._models.append(("CPU", model))

        die_names = [name for name, _ in self._models]
        print(
            f"[Whisper] {len(self._models)} die(s): {', '.join(die_names)}",
            file=sys.stderr,
        )

    def _load_optimum_model(self) -> Any:
        """Load OVModelForSpeechSeq2Seq on CPU."""
        model_dir = str(self.model_path)
        try:
            return OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir, device="CPU", ov_config={"CACHE_DIR": ""},
            )
        except Exception as e:
            logger.info("Default file names failed (%s), trying detected names", e)

        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            device="CPU",
            ov_config={"CACHE_DIR": ""},
            encoder_file_name=self.encoder_path.name,
            decoder_file_name=self.decoder_path.name,
        )

    def _patch_encoder_device(self, model: Any, die: str) -> None:
        """Recompile encoder on the target accelerator die."""
        import openvino as ov

        core = ov.Core()
        encoder_model = core.read_model(str(self.encoder_path))

        for inp in encoder_model.inputs:
            if inp.get_partial_shape().is_dynamic:
                encoder_model.reshape({inp.get_any_name(): [1, 128, 3000]})
                break

        ov_config: Dict[str, Any] = {"CACHE_DIR": ""}
        if hasattr(self.config, "openvino") and hasattr(self.config.openvino, "device_properties"):
            for key, value in self.config.openvino.device_properties.items():
                ov_config[key] = value

        model.encoder.request = core.compile_model(encoder_model, die, ov_config)

    # -- Sample processing --------------------------------------------------

    def _process_sample(self, sample_idx: int, model: Any) -> str:
        """Transcribe a single audio sample."""
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

    # -- LoadGen query dispatch ---------------------------------------------

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        """Offline: distribute samples across dies and run in parallel."""
        total = len(query_samples)
        num_dies = len(self._models)
        self._start_time = time.time()

        print(f"[Offline] {total} samples, {num_dies} die(s)", file=sys.stderr)

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        # Round-robin distribution
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

        print("[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = [
                pool.submit(_die_worker, idx, batch)
                for idx, batch in enumerate(die_batches)
                if batch
            ]
            while True:
                done = sum(1 for f in futures if f.done())
                completed = self._sample_count
                target = int(completed * 10 / total) if total else 10
                while dots_printed < target:
                    print(".", end="", file=sys.stderr, flush=True)
                    dots_printed += 1
                if done == len(futures):
                    break
                time.sleep(0.1)

            for f in futures:
                f.result()

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(
            f" {total}/{total} ({elapsed:.1f}s, {total / elapsed:.1f} qps)",
            file=sys.stderr,
        )

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
        """Single-die Offline path."""
        total = len(query_samples)
        responses = []
        arrays = []

        print("[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

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

            target = int(self._sample_count * 10 / total) if total else 10
            while dots_printed < target:
                print(".", end="", file=sys.stderr, flush=True)
                dots_printed += 1

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(
            f" {total}/{total} ({elapsed:.1f}s, {total / elapsed:.1f} qps)",
            file=sys.stderr,
        )

        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        """Server: round-robin across dies, respond per query."""
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

    # -- LoadGen integration ------------------------------------------------

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
