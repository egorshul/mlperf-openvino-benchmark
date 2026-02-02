"""Whisper ASR System Under Test implementations."""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from ..backends.base import BaseBackend
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
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoProcessor

        logger.debug("Loading Whisper model from %s", self.model_path)

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            logger.debug("Falling back to openai/whisper-large-v3 processor")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_path, ov_config={"CACHE_DIR": ""}, compile=True,
        )

    def flush_queries(self) -> None:
        pass

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
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            self._issue_query_server(query_samples)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        self._start_time = time.time()
        responses = []
        response_arrays = []

        print(f"[Offline] {total} samples", file=sys.stderr)

        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

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
                self.qsl.unload_query_samples,
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
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

        self._tokenizer = None

    def flush_queries(self) -> None:
        pass

    def _discover_decoder_inputs(self) -> Dict[str, str]:
        input_names = self.decoder.input_names
        result = {}

        for name in input_names:
            name_lower = name.lower()
            if "input_id" in name_lower or "decoder_input" in name_lower:
                result["input_ids"] = name
            elif "encoder_hidden" in name_lower or "encoder_output" in name_lower:
                result["encoder_hidden_states"] = name
            elif "attention_mask" in name_lower and "encoder" not in name_lower:
                result["attention_mask"] = name
            elif "encoder_attention_mask" in name_lower:
                result["encoder_attention_mask"] = name

        if "input_ids" not in result:
            for name in input_names:
                if "id" in name.lower():
                    result["input_ids"] = name
                    break

        if "encoder_hidden_states" not in result:
            for name in input_names:
                if "encoder" in name.lower() and "mask" not in name.lower():
                    result["encoder_hidden_states"] = name
                    break

        return result

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            from transformers import WhisperTokenizer
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
        except ImportError:
            logger.warning("transformers not installed, using basic token decoding")
            self._tokenizer = None

    def _decode_tokens(self, token_ids: List[int]) -> str:
        self._load_tokenizer()
        if self._tokenizer is not None:
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
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

        if "input_ids" in self._decoder_input_names:
            inputs[self._decoder_input_names["input_ids"]] = decoder_input_ids
        else:
            inputs["decoder_input_ids"] = decoder_input_ids

        if "encoder_hidden_states" in self._decoder_input_names:
            inputs[self._decoder_input_names["encoder_hidden_states"]] = encoder_hidden_states
        else:
            inputs["encoder_hidden_states"] = encoder_hidden_states

        if "attention_mask" in self._decoder_input_names:
            attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[self._decoder_input_names["attention_mask"]] = attn_mask

        if "encoder_attention_mask" in self._decoder_input_names:
            batch_size = encoder_hidden_states.shape[0]
            seq_len = encoder_hidden_states.shape[1]
            enc_attn_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            inputs[self._decoder_input_names["encoder_attention_mask"]] = enc_attn_mask

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
        _, text = self._generate(mel_features)
        return text

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            self._issue_query_server(query_samples)

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        total = len(query_samples)
        self._start_time = time.time()
        responses = []
        response_arrays = []

        print(f"[Offline] {total} samples", file=sys.stderr)

        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        for sample in query_samples:
            sample_idx = sample.index
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array("B", response_data.tobytes())
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, bi[0], bi[1])])

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
                self.qsl.unload_query_samples,
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
        self._start_time = 0.0

        self._sut_handle = None
        self._qsl_handle = None

    def flush_queries(self) -> None:
        pass

    def issue_queries(self, query_samples: List[Any]) -> None:
        self._query_count += len(query_samples)
        total = len(query_samples)
        self._start_time = time.time()
        responses = []
        response_arrays = []

        print(f"[Offline] {total} samples", file=sys.stderr)

        for sample in query_samples:
            sample_idx = sample.index
            features = self.qsl.get_features(sample_idx)
            mel_features = features["input_features"]

            inputs = {self.backend.input_names[0]: mel_features}
            outputs = self.backend.predict(inputs)
            encoder_output = list(outputs.values())[0]

            self._predictions[sample_idx] = encoder_output
            self._sample_count += 1

            response_array = array.array("B", encoder_output.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))

            _print_progress(self._sample_count, total, self._start_time)

        _print_progress(total, total, self._start_time)
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
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        return self._predictions.copy()

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
