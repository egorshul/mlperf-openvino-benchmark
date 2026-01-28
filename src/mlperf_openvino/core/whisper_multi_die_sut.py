"""Whisper SUT for multi-die NPU accelerators.

This module provides a Python SUT implementation for Whisper model
optimized for multi-die NPU configurations. It uses OpenVINO directly
(without optimum) for fine-grained control over model compilation
and inference distribution across dies.
"""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    ov = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    from transformers import WhisperTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    WhisperTokenizer = None

from .config import BenchmarkConfig, Scenario
from ..backends.device_discovery import discover_accelerator_devices
from ..datasets.librispeech import LibriSpeechQSL

logger = logging.getLogger(__name__)


# Whisper special tokens
SOT_TOKEN = 50258  # Start of transcript
EOT_TOKEN = 50257  # End of transcript
TRANSCRIBE_TOKEN = 50359  # Transcribe task
NO_TIMESTAMPS_TOKEN = 50363  # No timestamps
EN_TOKEN = 50259  # English language


class WhisperMultiDieSUT:
    """
    Whisper SUT for multi-die NPU accelerators.

    Distributes encoder inference across multiple NPU dies for improved throughput.
    Decoder inference uses KV-cache for efficient autoregressive generation.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
    ):
        """
        Initialize Whisper multi-die SUT.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO Whisper model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is required for multi-die inference")

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        # OpenVINO core
        self._core = ov.Core()

        # Device configuration
        self._device_prefix = config.openvino.get_device_prefix()
        self._target_devices: List[str] = []
        self._discover_devices()

        # Compiled models per die
        self._encoders: Dict[str, ov.CompiledModel] = {}
        self._decoders: Dict[str, ov.CompiledModel] = {}

        # Inference requests per die
        self._encoder_requests: Dict[str, ov.InferRequest] = {}
        self._decoder_requests: Dict[str, ov.InferRequest] = {}

        # Model info
        self._encoder_input_name: str = ""
        self._decoder_input_names: Dict[str, str] = {}
        self._use_kv_cache = False

        # Tokenizer
        self._tokenizer = None

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Accuracy mode
        self._is_accuracy_mode = False
        self._is_loaded = False

    def _discover_devices(self) -> None:
        """Discover available NPU dies."""
        if self.config.openvino.is_specific_die():
            # User specified a single die
            self._target_devices = [self.config.openvino.device]
        else:
            # Discover all dies
            self._target_devices = discover_accelerator_devices(
                self._core, self._device_prefix
            )

        if not self._target_devices:
            raise RuntimeError(
                f"No {self._device_prefix} devices found. "
                f"Available: {self._core.available_devices}"
            )

        logger.info(f"Discovered {len(self._target_devices)} {self._device_prefix} dies: {self._target_devices}")

    def _find_model_files(self) -> Tuple[Path, Path]:
        """Find encoder and decoder model files."""
        encoder_candidates = [
            self.model_path / "encoder_model.xml",
            self.model_path / "openvino_encoder_model.xml",
        ]

        decoder_candidates = [
            self.model_path / "decoder_with_past_model.xml",
            self.model_path / "openvino_decoder_with_past_model.xml",
            self.model_path / "decoder_model_merged.xml",
            self.model_path / "openvino_decoder_model_merged.xml",
            self.model_path / "decoder_model.xml",
            self.model_path / "openvino_decoder_model.xml",
        ]

        encoder_path = None
        decoder_path = None

        for ep in encoder_candidates:
            if ep.exists():
                encoder_path = ep
                break

        for dp in decoder_candidates:
            if dp.exists():
                decoder_path = dp
                # Check if using KV-cache version
                if "with_past" in dp.name or "merged" in dp.name:
                    self._use_kv_cache = True
                break

        if not encoder_path:
            raise FileNotFoundError(
                f"Encoder model not found in {self.model_path}. "
                f"Tried: {[p.name for p in encoder_candidates]}"
            )

        if not decoder_path:
            raise FileNotFoundError(
                f"Decoder model not found in {self.model_path}. "
                f"Tried: {[p.name for p in decoder_candidates]}"
            )

        logger.info(f"Encoder: {encoder_path.name}")
        logger.info(f"Decoder: {decoder_path.name} (KV-cache: {self._use_kv_cache})")

        return encoder_path, decoder_path

    def _get_compile_properties(self) -> Dict[str, Any]:
        """Get compilation properties for NPU."""
        props = {"CACHE_DIR": ""}

        # Add user-specified properties
        if self.config.openvino.device_properties:
            props.update(self.config.openvino.device_properties)

        return props

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load and compile models on all dies."""
        if self._is_loaded:
            return

        self._is_accuracy_mode = is_accuracy_mode

        encoder_path, decoder_path = self._find_model_files()

        # Read models
        logger.info("Reading models...")
        encoder_model = self._core.read_model(str(encoder_path))
        decoder_model = self._core.read_model(str(decoder_path))

        # Get input names
        self._encoder_input_name = encoder_model.inputs[0].get_any_name()
        self._discover_decoder_inputs(decoder_model)

        # Compile on all dies
        compile_props = self._get_compile_properties()

        logger.info(f"Compiling models on {len(self._target_devices)} dies...")

        for device in self._target_devices:
            logger.info(f"  Compiling on {device}...")

            # Compile encoder
            self._encoders[device] = self._core.compile_model(
                encoder_model, device, compile_props
            )
            self._encoder_requests[device] = self._encoders[device].create_infer_request()

            # Compile decoder
            self._decoders[device] = self._core.compile_model(
                decoder_model, device, compile_props
            )
            self._decoder_requests[device] = self._decoders[device].create_infer_request()

        # Load tokenizer
        self._load_tokenizer()

        self._is_loaded = True
        logger.info(f"Models loaded on {len(self._target_devices)} dies")

    def _discover_decoder_inputs(self, model: "ov.Model") -> None:
        """Discover decoder input names from model."""
        for inp in model.inputs:
            name = inp.get_any_name()
            name_lower = name.lower()

            if 'input_id' in name_lower or 'decoder_input' in name_lower:
                self._decoder_input_names['input_ids'] = name
            elif 'encoder_hidden' in name_lower or 'encoder_output' in name_lower:
                self._decoder_input_names['encoder_hidden_states'] = name
            elif 'attention_mask' in name_lower and 'encoder' not in name_lower:
                self._decoder_input_names['attention_mask'] = name
            elif 'encoder_attention_mask' in name_lower:
                self._decoder_input_names['encoder_attention_mask'] = name

        logger.debug(f"Decoder inputs: {self._decoder_input_names}")

    def _load_tokenizer(self) -> None:
        """Load Whisper tokenizer."""
        if self._tokenizer is not None:
            return

        if TOKENIZER_AVAILABLE:
            try:
                # Try loading from model path first
                self._tokenizer = WhisperTokenizer.from_pretrained(str(self.model_path))
                logger.info("Loaded tokenizer from model path")
            except Exception:
                try:
                    self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
                    logger.info("Loaded tokenizer from HuggingFace")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")
        else:
            logger.warning("transformers not installed, using basic token decoding")

    def warmup(self, iterations: int = 2) -> None:
        """Run warmup inferences on all dies."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded, call load() first")

        print(f"[Warmup] Whisper on {len(self._target_devices)} dies...", end="", file=sys.stderr, flush=True)

        # Create dummy input
        dummy_mel = np.zeros((1, 80, 3000), dtype=np.float32)

        for i in range(iterations):
            for device in self._target_devices:
                # Warmup encoder
                self._encoder_requests[device].infer({self._encoder_input_name: dummy_mel})

                # Warmup decoder (just one step)
                encoder_out = self._encoder_requests[device].get_output_tensor(0).data
                decoder_input = np.array([[SOT_TOKEN]], dtype=np.int64)

                decoder_inputs = {
                    self._decoder_input_names.get('input_ids', 'decoder_input_ids'): decoder_input,
                }
                if 'encoder_hidden_states' in self._decoder_input_names:
                    decoder_inputs[self._decoder_input_names['encoder_hidden_states']] = encoder_out

                self._decoder_requests[device].infer(decoder_inputs)

            print(".", end="", file=sys.stderr, flush=True)

        print(f" done", file=sys.stderr)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_dies(self) -> int:
        return len(self._target_devices)

    @property
    def name(self) -> str:
        return f"WhisperMultiDie-{self.num_dies}dies"

    def _encode(self, mel_features: np.ndarray, device: str) -> np.ndarray:
        """Run encoder on specified device."""
        request = self._encoder_requests[device]
        request.infer({self._encoder_input_name: mel_features})
        return request.get_output_tensor(0).data.copy()

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            # Filter special tokens
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            return f"[tokens: {len(token_ids)}]"

    def _generate(
        self,
        encoder_hidden_states: np.ndarray,
        device: str,
    ) -> Tuple[List[int], str]:
        """Generate transcript from encoder hidden states."""
        request = self._decoder_requests[device]

        # Initialize decoder input
        decoder_input = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]
        generated_tokens = []

        for step in range(self.max_new_tokens):
            decoder_input_ids = np.array([decoder_input], dtype=np.int64)

            # Build decoder inputs
            inputs = {
                self._decoder_input_names.get('input_ids', 'decoder_input_ids'): decoder_input_ids,
            }
            if 'encoder_hidden_states' in self._decoder_input_names:
                inputs[self._decoder_input_names['encoder_hidden_states']] = encoder_hidden_states

            if 'attention_mask' in self._decoder_input_names:
                attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
                inputs[self._decoder_input_names['attention_mask']] = attn_mask

            if 'encoder_attention_mask' in self._decoder_input_names:
                batch_size = encoder_hidden_states.shape[0]
                seq_len = encoder_hidden_states.shape[1]
                enc_attn_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                inputs[self._decoder_input_names['encoder_attention_mask']] = enc_attn_mask

            # Run decoder
            request.infer(inputs)
            logits = request.get_output_tensor(0).data

            # Get next token (greedy decoding)
            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                next_token_logits = logits[0, :]
            else:
                break

            next_token = int(np.argmax(next_token_logits))

            # Check for end of transcript
            if next_token == EOT_TOKEN:
                break

            generated_tokens.append(next_token)
            decoder_input.append(next_token)

        # Decode tokens to text
        text = self._decode_tokens(generated_tokens)
        return generated_tokens, text

    def _process_sample(self, sample_idx: int, device: str) -> str:
        """Process a single audio sample on specified device."""
        # Get mel features
        features = self.qsl.get_features(sample_idx)
        mel_features = features["input_features"]

        # Ensure correct shape
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, ...]

        if mel_features.dtype != np.float32:
            mel_features = mel_features.astype(np.float32)

        # Encode
        encoder_hidden_states = self._encode(mel_features, device)

        # Generate
        _, text = self._generate(encoder_hidden_states, device)

        return text

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries in Offline mode."""
        self._start_time = time.time()

        num_samples = len(query_samples)
        num_dies = len(self._target_devices)

        print(f"[Offline] Whisper: {num_samples} samples on {num_dies} dies", file=sys.stderr)

        responses = []
        response_arrays = []

        # Process samples, distributing across dies
        print(f"[Inference] ", end="", file=sys.stderr)
        dots_printed = 0

        for i, sample in enumerate(query_samples):
            sample_idx = sample.index
            self._sample_count += 1

            # Round-robin distribution across dies
            device = self._target_devices[i % num_dies]

            # Process sample
            text = self._process_sample(sample_idx, device)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            # Progress
            progress = int((i + 1) * 10 / num_samples)
            while dots_printed < progress:
                print(".", end="", file=sys.stderr, flush=True)
                dots_printed += 1

        while dots_printed < 10:
            print(".", end="", file=sys.stderr, flush=True)
            dots_printed += 1

        elapsed = time.time() - self._start_time
        print(f" {num_samples}/{num_samples} ({elapsed:.1f}s, {num_samples/elapsed:.1f} qps)", file=sys.stderr)

        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries in Server mode."""
        responses = []
        response_arrays = []

        num_dies = len(self._target_devices)

        for i, sample in enumerate(query_samples):
            sample_idx = sample.index
            self._sample_count += 1

            # Round-robin distribution
            device = self._target_devices[self._sample_count % num_dies]

            text = self._process_sample(sample_idx, device)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush pending queries."""
        pass

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._is_accuracy_mode = store

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


def is_whisper_multi_die_available() -> bool:
    """Check if Whisper multi-die SUT is available."""
    return OPENVINO_AVAILABLE and LOADGEN_AVAILABLE
