"""Whisper ASR System Under Test implementations."""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

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
from ..backends.base import BaseBackend
from ..datasets.librispeech import LibriSpeechQSL

logger = logging.getLogger(__name__)


def _generate_quiet(model, input_features, **kwargs):
    """Call model.generate() suppressing benign HF warnings."""
    import warnings

    tf_logger = logging.getLogger("transformers")
    prev_level = tf_logger.level

    tf_logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*attention.mask.*pad token.*")
        warnings.filterwarnings("ignore", message=r".*LogitsProcessor.*has been passed.*")
        warnings.filterwarnings("ignore", message=r".*generation_config.*default values.*modified.*")
        try:
            return model.generate(input_features, **kwargs)
        finally:
            tf_logger.setLevel(prev_level)


class WhisperOptimumSUT:
    """Whisper ASR SUT using Optimum-Intel OVModelForSpeechSeq2Seq."""

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "Optimum-Intel is required for Whisper inference. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        self._sut_handle = None
        self._qsl_handle = None

        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoProcessor

        logger.info(f"Loading Whisper model from {self.model_path}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Could not load processor from model path: {e}")
            logger.info("Falling back to openai/whisper-large-v3 processor")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            ov_config={"CACHE_DIR": ""},
            compile=True,
        )

        logger.info("Whisper model loaded successfully")

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        if self._progress_bar is not None:
            self._close_progress()

    def _process_sample(self, sample_idx: int) -> str:
        import torch

        features = self.qsl.get_features(sample_idx)
        input_features = features["input_features"]

        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)

        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        generated_ids = _generate_quiet(
            self.model, input_features,
            max_new_tokens=self.max_new_tokens,
            language="en",
            task="transcribe",
        )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper Offline inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc="Whisper Server inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperSUT:
    """Whisper ASR SUT with manual encoder-decoder inference (fallback)."""

    SOT_TOKEN = 50258
    EOT_TOKEN = 50257
    TRANSCRIBE_TOKEN = 50359
    NO_TIMESTAMPS_TOKEN = 50363
    EN_TOKEN = 50259

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_backend: BaseBackend,
        decoder_backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.encoder = encoder_backend
        self.decoder = decoder_backend
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        self._decoder_input_names = self._discover_decoder_inputs()

        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        self._sut_handle = None
        self._qsl_handle = None

        self._tokenizer = None

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        if self._progress_bar is not None:
            self._close_progress()

    def _discover_decoder_inputs(self) -> Dict[str, str]:
        input_names = self.decoder.input_names
        result = {}

        for name in input_names:
            name_lower = name.lower()
            if 'input_id' in name_lower or 'decoder_input' in name_lower:
                result['input_ids'] = name
            elif 'encoder_hidden' in name_lower or 'encoder_output' in name_lower:
                result['encoder_hidden_states'] = name
            elif 'attention_mask' in name_lower and 'encoder' not in name_lower:
                result['attention_mask'] = name
            elif 'encoder_attention_mask' in name_lower:
                result['encoder_attention_mask'] = name

        if 'input_ids' not in result:
            for name in input_names:
                if 'id' in name.lower():
                    result['input_ids'] = name
                    break

        if 'encoder_hidden_states' not in result:
            for name in input_names:
                if 'encoder' in name.lower() and 'mask' not in name.lower():
                    result['encoder_hidden_states'] = name
                    break

        return result

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            from transformers import WhisperTokenizer
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            logger.info("Loaded Whisper tokenizer")
        except ImportError:
            logger.warning("transformers not installed, using basic token decoding")
            self._tokenizer = None

    def _decode_tokens(self, token_ids: List[int]) -> str:
        self._load_tokenizer()

        if self._tokenizer is not None:
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            return f"[tokens: {len(token_ids)}]"

    def _encode(self, mel_features: np.ndarray) -> np.ndarray:
        inputs = {self.encoder.input_names[0]: mel_features}
        outputs = self.encoder.predict(inputs)
        return list(outputs.values())[0]

    def _decode_step(
        self,
        encoder_hidden_states: np.ndarray,
        decoder_input_ids: np.ndarray,
    ) -> np.ndarray:
        inputs = {}

        if 'input_ids' in self._decoder_input_names:
            inputs[self._decoder_input_names['input_ids']] = decoder_input_ids
        else:
            inputs['decoder_input_ids'] = decoder_input_ids

        if 'encoder_hidden_states' in self._decoder_input_names:
            inputs[self._decoder_input_names['encoder_hidden_states']] = encoder_hidden_states
        else:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        if 'attention_mask' in self._decoder_input_names:
            attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[self._decoder_input_names['attention_mask']] = attn_mask

        if 'encoder_attention_mask' in self._decoder_input_names:
            batch_size = encoder_hidden_states.shape[0]
            seq_len = encoder_hidden_states.shape[1]
            enc_attn_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            inputs[self._decoder_input_names['encoder_attention_mask']] = enc_attn_mask

        outputs = self.decoder.predict(inputs)
        return list(outputs.values())[0]

    def _generate(
        self,
        mel_features: np.ndarray,
        temperature: float = 0.0,
    ) -> Tuple[List[int], str]:
        encoder_hidden_states = self._encode(mel_features)

        decoder_input = [
            self.SOT_TOKEN,
            self.EN_TOKEN,
            self.TRANSCRIBE_TOKEN,
            self.NO_TIMESTAMPS_TOKEN,
        ]

        generated_tokens = []

        for step in range(self.max_new_tokens):
            decoder_input_ids = np.array([decoder_input], dtype=np.int64)
            logits = self._decode_step(encoder_hidden_states, decoder_input_ids)

            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                next_token_logits = logits[0, :]
            else:
                break

            if temperature == 0.0:
                next_token = int(np.argmax(next_token_logits))
            else:
                probs = np.exp(next_token_logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))

            if next_token == self.EOT_TOKEN:
                break

            generated_tokens.append(next_token)
            decoder_input.append(next_token)

        text = self._decode_tokens(generated_tokens)
        return generated_tokens, text

    def _process_sample(self, sample_idx: int) -> str:
        features = self.qsl.get_features(sample_idx)
        mel_features = features["input_features"]
        tokens, text = self._generate(mel_features)
        return text

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper Offline inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc="Whisper Server inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperEncoderOnlySUT:
    """Whisper encoder-only SUT for benchmarking encoder performance."""

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario

        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        self._sut_handle = None
        self._qsl_handle = None

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        if self._progress_bar is not None:
            self._close_progress()

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper encoder inference")

        responses = []
        response_arrays = []

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            features = self.qsl.get_features(sample_idx)
            mel_features = features["input_features"]

            inputs = {self.backend.input_names[0]: mel_features}
            outputs = self.backend.predict(inputs)
            encoder_output = list(outputs.values())[0]

            self._predictions[sample_idx] = encoder_output

            response_array = array.array('B', encoder_output.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


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
        import re

        core = ov.Core()
        devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")

        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        device_dies = [d for d in devices if pattern.match(d)]
        return sorted(device_dies)

    def _setup_models(self) -> None:
        import re
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
        import os

        model_dir = str(self.model_path)

        self._ensure_optimum_symlinks()

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

    def _ensure_optimum_symlinks(self) -> None:
        import os

        standard_map = {
            "openvino_encoder_model": self.encoder_path,
            "openvino_decoder_model": self.decoder_path,
        }

        for standard_stem, actual_path in standard_map.items():
            for ext in (".xml", ".bin"):
                standard_file = self.model_path / f"{standard_stem}{ext}"
                actual_file = actual_path.with_suffix(ext)
                if standard_file.exists() or not actual_file.exists():
                    continue
                try:
                    os.symlink(actual_file.resolve(), standard_file)
                    logger.info(f"Symlinked {standard_file.name} -> {actual_file.name}")
                except OSError as e:
                    logger.warning(f"Cannot create symlink {standard_file.name}: {e}")

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
        from concurrent.futures import ThreadPoolExecutor, as_completed

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
