"""Whisper ASR Multi-Die System Under Test implementation."""

import array
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Whisper ASR SUT for multi-die accelerators."""

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
        self._query_count = 0
        self._sample_count = 0
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
        devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")

        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        device_dies = [d for d in devices if pattern.match(d)]
        return sorted(device_dies)

    def _setup_models(self) -> None:
        from transformers import AutoProcessor

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for WhisperMultiDieSUT. "
                "Install with: pip install optimum[openvino]"
            )

        target_device = self.config.openvino.device if hasattr(self.config, 'openvino') else "CPU"

        logger.info(f"Setting up Whisper Multi-Die SUT (encoder={target_device}, decoder=CPU)")

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
                logger.warning(f"No {target_device} dies found, falling back to CPU")
                device_dies = ["CPU"]

        for die in device_dies:
            logger.info(f"Loading OVModelForSpeechSeq2Seq for {die}")
            try:
                model = self._load_optimum_model()

                if die != "CPU":
                    self._patch_encoder_device(model, die)

                self._models.append((die, model))
            except Exception as e:
                logger.warning(f"Failed to create model for {die}: {e}")

        if not self._models:
            logger.info("Creating CPU fallback model")
            model = self._load_optimum_model()
            self._models.append(("CPU", model))

        encoder_devices = [die for die, _ in self._models]
        print(f"[Setup] {len(self._models)} die(s), encoder={encoder_devices}, decoder=CPU",
              file=sys.stderr)

        for die_name, mdl in self._models:
            enc_tag = dec_tag = "?"
            try:
                exec_devs = mdl.encoder.request.get_property("EXECUTION_DEVICES")
                on_target = any(die_name.split(".")[0] in d for d in exec_devs)
                enc_tag = "OK" if on_target else f"WARN:CPU ({exec_devs})"
            except Exception:
                enc_tag = "unknown"
            try:
                exec_devs = mdl.decoder.request.get_property("EXECUTION_DEVICES")
                dec_tag = ",".join(exec_devs) if exec_devs else "unknown"
            except Exception:
                dec_tag = "CPU"
            print(f"  {die_name}: encoder=[{enc_tag}] decoder=[{dec_tag}]",
                  file=sys.stderr)

    def _load_optimum_model(self):
        model_dir = str(self.model_path)

        try:
            return OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                device="CPU",
                ov_config={"CACHE_DIR": ""},
            )
        except Exception as e:
            logger.info(f"Default file names failed ({e}), trying detected names")

        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            device="CPU",
            ov_config={"CACHE_DIR": ""},
            encoder_file_name=self.encoder_path.name,
            decoder_file_name=self.decoder_path.name,
        )

    def _patch_encoder_device(self, model, die: str) -> None:
        import openvino as ov

        core = ov.Core()

        encoder_model = core.read_model(str(self.encoder_path))

        for inp in encoder_model.inputs:
            if inp.get_partial_shape().is_dynamic:
                input_name = inp.get_any_name()
                encoder_model.reshape({input_name: [1, 128, 3000]})
                break

        ov_config = {"CACHE_DIR": ""}
        if hasattr(self.config, 'openvino') and hasattr(self.config.openvino, 'device_properties'):
            for key, value in self.config.openvino.device_properties.items():
                ov_config[key] = value

        compiled_encoder = core.compile_model(encoder_model, die, ov_config)

        try:
            exec_devices = compiled_encoder.get_property("EXECUTION_DEVICES")
            device_prefix = die.split(".")[0]
            on_target = any(device_prefix in d for d in exec_devices)
            if not on_target:
                logger.warning(f"Encoder requested {die} but EXECUTION_DEVICES={exec_devices}")
        except Exception:
            pass

        model.encoder.request = compiled_encoder

    def _get_next_model(self) -> Tuple[str, Any]:
        die_name, model = self._models[self._model_index]
        self._model_index = (self._model_index + 1) % len(self._models)
        return die_name, model

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

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

        print(f"[Offline] {total} samples, {num_dies} die(s)", file=sys.stderr)

        if num_dies <= 1:
            self._issue_queries_offline_sequential(query_samples)
            return

        die_batches: List[List[Tuple[Any, int]]] = [[] for _ in range(num_dies)]
        for i, sample in enumerate(query_samples):
            die_batches[i % num_dies].append((sample, sample.index))

        batch_counts = [len(b) for b in die_batches]
        print(f"[Distribute] {batch_counts} samples per die", file=sys.stderr)

        def _worker(die_idx: int, batch):
            _, model = self._models[die_idx]
            results = []
            for sample, sample_idx in batch:
                text = self._process_sample(sample_idx, model)
                results.append((sample, sample_idx, text))
            return results

        all_results: List[Tuple[Any, int, str]] = []
        print(f"[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = {
                pool.submit(_worker, die_idx, batch): die_idx
                for die_idx, batch in enumerate(die_batches)
                if batch
            }
            for future in as_completed(futures):
                for sample, sample_idx, text in future.result():
                    self._predictions[sample_idx] = text
                    self._sample_count += 1

                    progress = int(self._sample_count * 10 / total)
                    while dots_printed < progress:
                        print(".", end="", file=sys.stderr, flush=True)
                        dots_printed += 1

                all_results.extend(future.result())

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(f" {total}/{total} ({elapsed:.1f}s, {total/elapsed:.1f} qps)",
              file=sys.stderr)

        responses = []
        response_arrays = []
        for sample, _idx, text in sorted(all_results, key=lambda r: r[1]):
            text_bytes = text.encode('utf-8')
            response_array = array.array('B', text_bytes)
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)

    def _issue_queries_offline_sequential(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        responses = []
        response_arrays = []

        print(f"[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        for sample in query_samples:
            sample_idx = sample.index
            die_name, model = self._get_next_model()

            text = self._process_sample(sample_idx, model)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            text_bytes = text.encode('utf-8')
            response_array = array.array('B', text_bytes)
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            if total > 0:
                progress = int(self._sample_count * 10 / total)
                while dots_printed < progress:
                    print(".", end="", file=sys.stderr, flush=True)
                    dots_printed += 1

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(f" {total}/{total} ({elapsed:.1f}s, {total/elapsed:.1f} qps)",
              file=sys.stderr)

        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index

            die_name, model = self._get_next_model()

            text = self._process_sample(sample_idx, model)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            text_bytes = text.encode('utf-8')
            response_array = array.array('B', text_bytes)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self) -> None:
        pass

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples)
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._model_index = 0
