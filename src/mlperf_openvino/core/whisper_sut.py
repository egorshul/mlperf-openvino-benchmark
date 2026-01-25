"""
Whisper-specific System Under Test implementation.

This module provides SUT implementation optimized for Whisper ASR model,
using optimum-intel OVModelForSpeechSeq2Seq for proper encoder-decoder inference.
"""

import array
import logging
import sys
import threading
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

# Optimum-Intel for proper Whisper inference
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


class WhisperOptimumSUT:
    """
    System Under Test for Whisper ASR using Optimum-Intel.

    Uses OVModelForSpeechSeq2Seq for proper encoder-decoder inference
    with correct KV-cache handling and token generation.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,  # Leave room for special tokens (448 - 8)
    ):
        """
        Initialize Whisper SUT using Optimum-Intel.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO Whisper model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
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

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Load model and processor
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model using Optimum-Intel."""
        from transformers import AutoProcessor

        logger.info(f"Loading Whisper model from {self.model_path}")

        # Load processor (tokenizer + feature extractor)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Could not load processor from model path: {e}")
            logger.info("Falling back to openai/whisper-large-v3 processor")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Load OpenVINO model (exported with --task automatic-speech-recognition-with-past)
        ov_config = {"CACHE_DIR": ""}
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            ov_config=ov_config,
            compile=True,
        )

        logger.info("Whisper model loaded successfully")

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
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
        """Update progress by n samples."""
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
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def _process_sample(self, sample_idx: int) -> str:
        """
        Process a single audio sample.

        Args:
            sample_idx: Sample index

        Returns:
            Transcribed text
        """
        import torch

        # Get preprocessed mel features from QSL
        features = self.qsl.get_features(sample_idx)
        input_features = features["input_features"]

        # Convert to tensor
        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)

        # Ensure correct shape (batch, n_mels, time)
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        # Generate transcription using model with KV-cache support
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=self.max_new_tokens,
            language="en",
            task="transcribe",
        )

        # Decode tokens to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
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
        """Process queries for Server scenario."""
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
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperSUT:
    """
    System Under Test for Whisper ASR model (fallback implementation).

    Uses manual encoder-decoder inference when optimum-intel is not available.
    Prefer WhisperOptimumSUT when possible.
    """

    # Whisper special tokens
    SOT_TOKEN = 50258  # Start of transcript
    EOT_TOKEN = 50257  # End of transcript
    TRANSCRIBE_TOKEN = 50359  # Transcribe task
    NO_TIMESTAMPS_TOKEN = 50363  # No timestamps
    EN_TOKEN = 50259  # English language

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_backend: BaseBackend,
        decoder_backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,  # Leave room for special tokens (448 - 8)
    ):
        """
        Initialize Whisper SUT.

        Args:
            config: Benchmark configuration
            encoder_backend: OpenVINO backend for encoder
            decoder_backend: OpenVINO backend for decoder
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.encoder = encoder_backend
        self.decoder = decoder_backend
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        # Discover decoder input names
        self._decoder_input_names = self._discover_decoder_inputs()

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Tokenizer for decoding (lazy loaded)
        self._tokenizer = None
    
    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
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
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            # Simple text-based progress update
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Close progress bar if still open (for Server mode)
        if self._progress_bar is not None:
            self._close_progress()

    def _discover_decoder_inputs(self) -> Dict[str, str]:
        """
        Discover decoder input names from the model.

        Returns:
            Dictionary mapping semantic names to actual input names:
            - 'input_ids': name for decoder input IDs
            - 'encoder_hidden_states': name for encoder output
            - 'attention_mask': name for attention mask (optional)
        """
        input_names = self.decoder.input_names
        result = {}

        # Find input_ids (decoder_input_ids or input_ids)
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

        # Fallback if not found
        if 'input_ids' not in result:
            # Try first input that looks like IDs
            for name in input_names:
                if 'id' in name.lower():
                    result['input_ids'] = name
                    break

        if 'encoder_hidden_states' not in result:
            # Try to find encoder output
            for name in input_names:
                if 'encoder' in name.lower() and 'mask' not in name.lower():
                    result['encoder_hidden_states'] = name
                    break

        return result

    def _load_tokenizer(self):
        """Load Whisper tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import WhisperTokenizer
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            logger.info("Loaded Whisper tokenizer")
        except ImportError:
            logger.warning(
                "transformers not installed, using basic token decoding. "
                "Install with: pip install transformers"
            )
            self._tokenizer = None
    
    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        self._load_tokenizer()
        
        if self._tokenizer is not None:
            # Filter special tokens
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            # Basic fallback - just return token IDs as string
            return f"[tokens: {len(token_ids)}]"
    
    def _encode(self, mel_features: np.ndarray) -> np.ndarray:
        """
        Run encoder on mel spectrogram.
        
        Args:
            mel_features: Mel spectrogram of shape (batch, n_mels, time)
            
        Returns:
            Encoder hidden states
        """
        inputs = {self.encoder.input_names[0]: mel_features}
        outputs = self.encoder.predict(inputs)
        return list(outputs.values())[0]
    
    def _decode_step(
        self,
        encoder_hidden_states: np.ndarray,
        decoder_input_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Run one decoder step.

        Args:
            encoder_hidden_states: Encoder output
            decoder_input_ids: Current token sequence

        Returns:
            Logits for next token
        """
        inputs = {}

        # Use discovered input names
        if 'input_ids' in self._decoder_input_names:
            inputs[self._decoder_input_names['input_ids']] = decoder_input_ids
        else:
            inputs['decoder_input_ids'] = decoder_input_ids

        if 'encoder_hidden_states' in self._decoder_input_names:
            inputs[self._decoder_input_names['encoder_hidden_states']] = encoder_hidden_states
        else:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        # Add attention masks if required by model
        if 'attention_mask' in self._decoder_input_names:
            # Create attention mask (all ones for valid tokens)
            attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[self._decoder_input_names['attention_mask']] = attn_mask

        if 'encoder_attention_mask' in self._decoder_input_names:
            # Create encoder attention mask
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
        """
        Generate transcript from mel spectrogram.

        Args:
            mel_features: Mel spectrogram
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Tuple of (token_ids, decoded_text)
        """
        # Encode audio
        encoder_hidden_states = self._encode(mel_features)

        # Initialize decoder input with special tokens
        # [SOT, language, task, no_timestamps]
        decoder_input = [
            self.SOT_TOKEN,
            self.EN_TOKEN,
            self.TRANSCRIBE_TOKEN,
            self.NO_TIMESTAMPS_TOKEN,
        ]

        generated_tokens = []

        for step in range(self.max_new_tokens):
            # Prepare decoder input
            decoder_input_ids = np.array([decoder_input], dtype=np.int64)

            # Get logits
            logits = self._decode_step(encoder_hidden_states, decoder_input_ids)

            # Get next token (greedy or sampling)
            # Logits shape should be (batch, seq_len, vocab_size)
            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                # Shape might be (batch, vocab_size) for last token only
                next_token_logits = logits[0, :]
            else:
                break

            if temperature == 0.0:
                next_token = int(np.argmax(next_token_logits))
            else:
                # Softmax with temperature
                probs = np.exp(next_token_logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))

            # Check for end of transcript
            if next_token == self.EOT_TOKEN:
                break

            generated_tokens.append(next_token)
            decoder_input.append(next_token)

        # Decode tokens to text
        text = self._decode_tokens(generated_tokens)

        return generated_tokens, text
    
    def _process_sample(self, sample_idx: int) -> str:
        """
        Process a single audio sample.

        Args:
            sample_idx: Sample index

        Returns:
            Transcribed text
        """
        features = self.qsl.get_features(sample_idx)
        mel_features = features["input_features"]
        tokens, text = self._generate(mel_features)
        return text
    
    def issue_queries(self, query_samples: List[Any]) -> None:
        """
        Process queries from LoadGen.
        
        Args:
            query_samples: List of QuerySample objects
        """
        self._query_count += len(query_samples)
        
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")
    
    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        # Start progress tracking
        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper Offline inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response (using dummy data for LoadGen)
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        # Close progress
        self._close_progress()

        lg.QuerySamplesComplete(responses)
    
    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        # Start progress tracking if first query
        if self._sample_count == 0:
            self._start_progress(0, desc="Whisper Server inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        lg.QuerySamplesComplete(responses)
    
    def get_sut(self) -> Any:
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries
            )
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperEncoderOnlySUT:
    """
    Simplified SUT that only runs the Whisper encoder.
    
    Useful for benchmarking encoder performance separately,
    or when using external decoder/beam search.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize encoder-only SUT.
        
        Args:
            config: Benchmark configuration
            backend: OpenVINO backend for encoder
            qsl: Query Sample Library
            scenario: MLPerf scenario
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")
        
        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario
        
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
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
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            # Simple text-based progress update
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Close progress bar if still open
        if self._progress_bar is not None:
            self._close_progress()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        # Start progress tracking
        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper encoder inference")

        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Get input features
            features = self.qsl.get_features(sample_idx)
            mel_features = features["input_features"]

            # Run encoder
            inputs = {self.backend.input_names[0]: mel_features}
            outputs = self.backend.predict(inputs)
            encoder_output = list(outputs.values())[0]

            self._predictions[sample_idx] = encoder_output

            # Create response - use array.array for safe memory handling
            response_array = array.array('B', encoder_output.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        # Close progress
        self._close_progress()

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries
            )
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all encoder outputs."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperNPUSUT:
    """
    System Under Test for Whisper ASR optimized for NPU accelerators.

    This implementation uses OpenVINO directly for maximum compatibility
    with NPU devices. It supports:
    - Multi-die NPU (compiles on all dies: NPU.0, NPU.1, ... when device=NPU)
    - Static shapes with sequence length buckets (required for NPU)
    - KV-cache for efficient autoregressive decoding (if available)
    - Round-robin request distribution across dies

    For MLPerf v5.1 Whisper benchmark:
    - Model: Whisper-Large-v3 (1.55B parameters)
    - Input: 128 mel bins, 3000 time frames (30 seconds)
    - Accuracy target: 97.9329% Word Accuracy
    """

    # Whisper special tokens (from tokenizer)
    SOT_TOKEN = 50258  # Start of transcript <|startoftranscript|>
    EOT_TOKEN = 50257  # End of transcript <|endoftext|>
    TRANSCRIBE_TOKEN = 50359  # <|transcribe|>
    NO_TIMESTAMPS_TOKEN = 50363  # <|notimestamps|>
    EN_TOKEN = 50259  # <|en|> English language
    PAD_TOKEN = 50257  # Use EOT as padding

    # Whisper-Large-v3 architecture constants
    NUM_DECODER_LAYERS = 32
    NUM_ATTENTION_HEADS = 20
    HEAD_DIM = 64  # 1280 / 20
    ENCODER_SEQ_LEN = 1500  # Encoder output sequence length
    ENCODER_HIDDEN_SIZE = 1280

    # Sequence length buckets for static decoder models
    SEQ_BUCKETS = [16, 32, 64, 128, 256, 448]

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        qsl: "LibriSpeechQSL",
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
        device: str = "NPU",
    ):
        """
        Initialize Whisper NPU SUT.

        Args:
            config: Benchmark configuration
            encoder_path: Path to OpenVINO encoder model (.xml)
            decoder_path: Path to OpenVINO decoder model (.xml)
            qsl: Query Sample Library
            scenario: MLPerf scenario (Offline or Server)
            max_new_tokens: Maximum tokens to generate (448 - 8 special tokens)
            device: OpenVINO device (NPU for all dies, NPU.0 for specific die)
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens
        self.device = device

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # OpenVINO core
        self._core = None

        # Multi-die support
        # For static shapes: {die_name: {"encoder": request, "decoder_buckets": {seq_len: request}}}
        self._die_contexts: Dict[str, Dict[str, Any]] = {}
        self._active_devices: List[str] = []
        self._request_counter = 0
        self._request_lock = threading.Lock()

        # Model info (shared across dies)
        self._encoder_input_name = None
        self._encoder_output_name = None
        self._decoder_inputs = {}

        # Static shapes mode
        self._use_static_shapes = False
        self._available_buckets: List[int] = []

        # KV-cache support
        self._has_kv_cache = False
        self._kv_cache_inputs = []
        self._kv_cache_outputs = []

        # Tokenizer (lazy loaded)
        self._tokenizer = None

        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load encoder and decoder models on all NPU dies."""
        import openvino as ov
        from ..backends.device_discovery import discover_accelerator_devices

        logger.info(f"Loading Whisper models for {self.device}")

        self._core = ov.Core()

        # NPU-optimized compilation config from -p parameters
        compile_config = {}
        if self.config.openvino.device_properties:
            compile_config.update(self.config.openvino.device_properties)
            logger.info(f"Compile properties: {compile_config}")
        else:
            logger.info("No custom compile properties specified")

        # Discover target devices
        device_prefix = self.config.openvino.get_device_prefix()

        if self.config.openvino.is_specific_die():
            self._active_devices = [self.device]
        else:
            self._active_devices = discover_accelerator_devices(self._core, device_prefix)

        if not self._active_devices:
            self._active_devices = [self.device]

        logger.info(f"Will use {len(self._active_devices)} die(s): {self._active_devices}")

        model_dir = self.encoder_path.parent

        # List all available models
        all_xml = list(model_dir.glob("*.xml"))
        logger.info(f"Available models in {model_dir}: {[f.name for f in all_xml]}")

        # Check for static shape models first (required for NPU without dynamic shape support)
        static_encoder = self._find_static_encoder(model_dir)
        static_decoders = self._find_static_decoders(model_dir)

        if static_encoder and static_decoders:
            logger.info(f"Found static encoder: {static_encoder.name}")
            logger.info(f"Found {len(static_decoders)} static decoder buckets: {list(static_decoders.keys())}")
            self._use_static_shapes = True
            self._available_buckets = sorted(static_decoders.keys())
            self._load_static_models(static_encoder, static_decoders, compile_config)
            logger.info(
                f"Static shapes mode enabled (NPU compatible). "
                f"Buckets: {self._available_buckets}, max seq: {self._available_buckets[-1]}"
            )
        else:
            logger.info("Static models not found, trying dynamic models...")
            logger.info(
                "To create static models for NPU, run:\n"
                "  python -c \"from mlperf_openvino.utils import export_whisper_for_npu; "
                "export_whisper_for_npu('./models', static_shapes=True)\""
            )
            self._use_static_shapes = False
            self._load_dynamic_models(compile_config)

    def _find_static_encoder(self, model_dir: Path) -> Optional[Path]:
        """Find static encoder model."""
        candidates = [
            model_dir / "encoder_static_b1.xml",
            model_dir / "openvino_encoder_static_b1.xml",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _find_static_decoders(self, model_dir: Path) -> Dict[int, Path]:
        """Find static decoder models for each sequence length bucket."""
        decoders = {}
        for seq_len in self.SEQ_BUCKETS:
            candidates = [
                model_dir / f"decoder_static_b1_s{seq_len}.xml",
                model_dir / f"openvino_decoder_static_b1_s{seq_len}.xml",
            ]
            for path in candidates:
                if path.exists():
                    decoders[seq_len] = path
                    break
        return decoders

    def _load_static_models(
        self,
        encoder_path: Path,
        decoder_paths: Dict[int, Path],
        compile_config: Dict
    ) -> None:
        """Load static shape models for NPU."""
        logger.info("Loading static shape models...")

        # Read encoder
        encoder_model = self._core.read_model(str(encoder_path))
        self._encoder_input_name = encoder_model.inputs[0].get_any_name()
        self._encoder_output_name = encoder_model.outputs[0].get_any_name()

        # Read first decoder to get input names
        first_decoder_path = list(decoder_paths.values())[0]
        first_decoder = self._core.read_model(str(first_decoder_path))
        self._decoder_inputs = self._discover_decoder_inputs(first_decoder)

        # Log decoder info
        all_inputs = [inp.get_any_name() for inp in first_decoder.inputs]
        all_outputs = [out.get_any_name() for out in first_decoder.outputs]
        logger.info(f"Decoder inputs: {all_inputs}")
        logger.info(f"Decoder outputs: {all_outputs}")

        # Compile for each die
        for die_name in self._active_devices:
            logger.info(f"Compiling static models for {die_name}...")

            die_ctx = {"encoder": None, "decoder_buckets": {}}

            # Compile encoder
            try:
                encoder_compiled = self._core.compile_model(encoder_model, die_name, compile_config)
                die_ctx["encoder"] = encoder_compiled.create_infer_request()
                logger.info(f"  {die_name}: encoder compiled")
            except Exception as e:
                raise RuntimeError(f"Failed to compile static encoder on {die_name}: {e}")

            # Compile decoders for each bucket
            for seq_len, decoder_path in decoder_paths.items():
                try:
                    decoder_model = self._core.read_model(str(decoder_path))
                    decoder_compiled = self._core.compile_model(decoder_model, die_name, compile_config)
                    die_ctx["decoder_buckets"][seq_len] = decoder_compiled.create_infer_request()
                    logger.info(f"  {die_name}: decoder (seq={seq_len}) compiled")
                except Exception as e:
                    logger.warning(f"Failed to compile decoder seq={seq_len} on {die_name}: {e}")

            self._die_contexts[die_name] = die_ctx

        logger.info(f"Static models loaded on {len(self._active_devices)} die(s)")

    def _load_dynamic_models(self, compile_config: Dict) -> None:
        """Load dynamic shape models (fallback, may not work on all NPUs)."""
        logger.info("Loading dynamic shape models...")

        # Read encoder
        encoder_model = self._core.read_model(str(self.encoder_path))

        # Try to find decoder_with_past for KV-cache
        decoder_with_past_path = self._find_decoder_with_past()
        if decoder_with_past_path:
            logger.info(f"Using decoder_with_past: {decoder_with_past_path}")
            decoder_model = self._core.read_model(str(decoder_with_past_path))
            actual_decoder_path = decoder_with_past_path
        else:
            logger.warning(
                f"decoder_with_past not found! Using {self.decoder_path} "
                "(may fail on NPU due to dynamic shapes)"
            )
            decoder_model = self._core.read_model(str(self.decoder_path))
            actual_decoder_path = self.decoder_path

        # Detect KV-cache support
        self._has_kv_cache = self._detect_kv_cache(decoder_model)
        if self._has_kv_cache:
            logger.info(f"KV-cache enabled: {len(self._kv_cache_inputs)} cache tensors")
        else:
            logger.info("KV-cache not available, using full sequence decoding")

        # Get input/output names
        self._encoder_input_name = encoder_model.inputs[0].get_any_name()
        self._encoder_output_name = encoder_model.outputs[0].get_any_name()
        self._decoder_inputs = self._discover_decoder_inputs(decoder_model)

        # Log decoder info
        all_inputs = [inp.get_any_name() for inp in decoder_model.inputs]
        all_outputs = [out.get_any_name() for out in decoder_model.outputs]
        logger.info(f"Decoder inputs: {all_inputs}")
        logger.info(f"Decoder outputs: {all_outputs}")

        # Compile for each die
        for die_name in self._active_devices:
            logger.info(f"Compiling dynamic models for {die_name}...")

            die_ctx = {"encoder": None, "decoder": None}

            try:
                encoder_compiled = self._core.compile_model(encoder_model, die_name, compile_config)
                die_ctx["encoder"] = encoder_compiled.create_infer_request()
                logger.info(f"  {die_name}: encoder compiled")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compile encoder on {die_name}: {e}\n"
                    f"Encoder path: {self.encoder_path}\n"
                    f"Compile config: {compile_config}"
                )

            try:
                decoder_compiled = self._core.compile_model(decoder_model, die_name, compile_config)
                die_ctx["decoder"] = decoder_compiled.create_infer_request()
                logger.info(f"  {die_name}: decoder compiled")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compile decoder on {die_name}: {e}\n"
                    f"Decoder path: {actual_decoder_path}\n"
                    f"Compile config: {compile_config}\n"
                    f"Hint: NPU may require static shape models. Run:\n"
                    f"  python -c \"from mlperf_openvino.utils import export_whisper_for_npu; "
                    f"export_whisper_for_npu('./models', static_shapes=True)\""
                )

            self._die_contexts[die_name] = die_ctx

        logger.info(f"Dynamic models loaded on {len(self._active_devices)} die(s)")

    def _get_next_die(self) -> str:
        """Get next die for round-robin distribution."""
        with self._request_lock:
            die = self._active_devices[self._request_counter % len(self._active_devices)]
            self._request_counter += 1
        return die

    @property
    def num_dies(self) -> int:
        """Number of active dies."""
        return len(self._active_devices)

    @property
    def active_devices(self) -> List[str]:
        """List of active device names."""
        return self._active_devices.copy()

    def _find_decoder_with_past(self) -> Optional[Path]:
        """Find decoder_with_past model for KV-cache support."""
        decoder_dir = self.decoder_path.parent

        # List all XML files for debugging
        all_xml = list(decoder_dir.glob("*.xml"))
        logger.info(f"Available models in {decoder_dir}: {[f.name for f in all_xml]}")

        # Priority order for decoder with KV-cache
        # Optimum-intel creates different names depending on version
        candidates = [
            # With "openvino_" prefix (newer optimum)
            decoder_dir / "openvino_decoder_with_past_model.xml",
            # Without prefix (older optimum)
            decoder_dir / "decoder_with_past_model.xml",
            # Merged decoder (has both with and without past)
            decoder_dir / "openvino_decoder_model_merged.xml",
            decoder_dir / "decoder_model_merged.xml",
            # Some versions use just "decoder" with past inside
            decoder_dir / "openvino_decoder.xml",
            decoder_dir / "decoder.xml",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    def _detect_kv_cache(self, decoder_model) -> bool:
        """Detect if decoder model has KV-cache inputs/outputs."""
        self._kv_cache_inputs = []
        self._kv_cache_outputs = []

        # Log all inputs/outputs for debugging
        all_inputs = [inp.get_any_name() for inp in decoder_model.inputs]
        all_outputs = [out.get_any_name() for out in decoder_model.outputs]
        logger.info(f"Decoder inputs: {all_inputs}")
        logger.info(f"Decoder outputs: {all_outputs}")

        # Patterns for KV-cache detection
        kv_input_patterns = ['past_key_value', 'past_key', 'past', 'cache', 'kv_cache']
        kv_output_patterns = ['present', 'past_key', 'new_past', 'cache']

        for inp in decoder_model.inputs:
            name = inp.get_any_name()
            name_lower = name.lower()
            if any(p in name_lower for p in kv_input_patterns):
                self._kv_cache_inputs.append(name)

        for out in decoder_model.outputs:
            name = out.get_any_name()
            name_lower = name.lower()
            if any(p in name_lower for p in kv_output_patterns):
                self._kv_cache_outputs.append(name)

        return len(self._kv_cache_inputs) > 0 and len(self._kv_cache_outputs) > 0

    def _discover_decoder_inputs(self, decoder_model) -> Dict[str, str]:
        """Discover decoder input tensor names."""
        result = {}

        for input_info in decoder_model.inputs:
            name = input_info.get_any_name()
            name_lower = name.lower()

            if 'input_id' in name_lower or 'decoder_input' in name_lower:
                result['input_ids'] = name
            elif 'encoder_hidden' in name_lower or 'encoder_output' in name_lower:
                result['encoder_hidden_states'] = name
            elif 'encoder_attention_mask' in name_lower:
                result['encoder_attention_mask'] = name
            elif 'attention_mask' in name_lower:
                result['attention_mask'] = name

        return result

    def _load_tokenizer(self) -> None:
        """Load Whisper tokenizer for decoding."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import WhisperTokenizer
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            logger.info("Loaded Whisper tokenizer")
        except ImportError:
            logger.warning("transformers not installed, using basic decoding")
            self._tokenizer = None

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        self._load_tokenizer()

        if self._tokenizer is not None:
            # Filter special tokens
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            return f"[tokens: {len(token_ids)}]"

    def _encode(self, mel_features: np.ndarray, die_name: Optional[str] = None) -> np.ndarray:
        """Run encoder on mel spectrogram.

        Args:
            mel_features: Mel spectrogram of shape (batch, n_mels, time)
            die_name: Specific die to use, or None for round-robin

        Returns:
            Encoder hidden states
        """
        if die_name is None:
            die_name = self._get_next_die()

        die_ctx = self._die_contexts[die_name]
        encoder_request = die_ctx["encoder"]
        encoder_request.infer({self._encoder_input_name: mel_features})
        return encoder_request.get_tensor(self._encoder_output_name).data.copy()

    def _get_decoder_bucket(self, seq_len: int) -> int:
        """Get the smallest bucket that fits the sequence length."""
        for bucket in self._available_buckets:
            if seq_len <= bucket:
                return bucket
        # Return largest bucket if sequence is too long
        return self._available_buckets[-1] if self._available_buckets else seq_len

    def _decode_step(
        self,
        encoder_hidden_states: np.ndarray,
        decoder_input_ids: np.ndarray,
        die_name: str,
    ) -> np.ndarray:
        """Run one decoder step.

        Args:
            encoder_hidden_states: Encoder output
            decoder_input_ids: Current token sequence
            die_name: Device die to use for inference

        Returns:
            Logits for next token
        """
        die_ctx = self._die_contexts[die_name]
        seq_len = decoder_input_ids.shape[1]

        # Select decoder based on mode (static buckets or dynamic)
        if self._use_static_shapes:
            bucket = self._get_decoder_bucket(seq_len)
            decoder_request = die_ctx["decoder_buckets"].get(bucket)
            if decoder_request is None:
                raise RuntimeError(
                    f"No decoder for bucket {bucket}. Available: {list(die_ctx['decoder_buckets'].keys())}"
                )
            # Pad input_ids to bucket size
            if seq_len < bucket:
                padded_ids = np.full((1, bucket), self.PAD_TOKEN, dtype=np.int64)
                padded_ids[0, :seq_len] = decoder_input_ids[0, :]
                decoder_input_ids = padded_ids
        else:
            decoder_request = die_ctx["decoder"]

        inputs = {}

        # Set decoder input IDs
        if 'input_ids' in self._decoder_inputs:
            inputs[self._decoder_inputs['input_ids']] = decoder_input_ids
        else:
            inputs['decoder_input_ids'] = decoder_input_ids

        # Set encoder hidden states
        if 'encoder_hidden_states' in self._decoder_inputs:
            inputs[self._decoder_inputs['encoder_hidden_states']] = encoder_hidden_states
        else:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        # Add attention masks if required
        actual_seq_len = decoder_input_ids.shape[1]
        if 'attention_mask' in self._decoder_inputs:
            if self._use_static_shapes:
                # For static shapes, mask out padding
                attn_mask = np.zeros((1, actual_seq_len), dtype=np.int64)
                attn_mask[0, :seq_len] = 1
            else:
                attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[self._decoder_inputs['attention_mask']] = attn_mask

        if 'encoder_attention_mask' in self._decoder_inputs:
            batch_size = encoder_hidden_states.shape[0]
            enc_seq_len = encoder_hidden_states.shape[1]
            enc_attn_mask = np.ones((batch_size, enc_seq_len), dtype=np.int64)
            inputs[self._decoder_inputs['encoder_attention_mask']] = enc_attn_mask

        decoder_request.infer(inputs)

        # Get first output (logits)
        logits = decoder_request.get_output_tensor(0).data.copy()

        # For static shapes, only return logits for non-padded positions
        if self._use_static_shapes and logits.ndim == 3 and logits.shape[1] > seq_len:
            logits = logits[:, :seq_len, :]

        return logits

    def _generate(self, mel_features: np.ndarray) -> Tuple[List[int], str]:
        """Generate transcript from mel spectrogram.

        Uses KV-cache if available for ~10x faster decoding.
        Uses round-robin die selection for multi-die parallelism.
        Each sample uses single die for both encoder and decoder.

        For static shapes mode (NPU): uses bucket-based decoding without KV-cache.

        Args:
            mel_features: Mel spectrogram of shape (batch, n_mels, time)

        Returns:
            Tuple of (token_ids, decoded_text)
        """
        # Select die for this sample (round-robin)
        die_name = self._get_next_die()

        # Encode audio on selected die
        encoder_hidden_states = self._encode(mel_features, die_name)

        # Initialize decoder with special tokens
        # [SOT, language, task, no_timestamps]
        initial_tokens = [
            self.SOT_TOKEN,
            self.EN_TOKEN,
            self.TRANSCRIBE_TOKEN,
            self.NO_TIMESTAMPS_TOKEN,
        ]

        # Static shapes mode doesn't support KV-cache (uses bucket-based decoding)
        if self._use_static_shapes:
            return self._generate_without_kv_cache(encoder_hidden_states, initial_tokens, die_name)
        elif self._has_kv_cache:
            return self._generate_with_kv_cache(encoder_hidden_states, initial_tokens, die_name)
        else:
            return self._generate_without_kv_cache(encoder_hidden_states, initial_tokens, die_name)

    def _generate_without_kv_cache(
        self,
        encoder_hidden_states: np.ndarray,
        initial_tokens: List[int],
        die_name: str,
    ) -> Tuple[List[int], str]:
        """Generate without KV-cache (slower, recomputes all attention).

        For static shapes mode, uses bucket-based decoder selection with padding.
        """
        decoder_input = initial_tokens.copy()
        generated_tokens = []

        # For static shapes, check maximum sequence length
        max_bucket = self._available_buckets[-1] if self._available_buckets else float('inf')

        for step in range(self.max_new_tokens):
            # Check if we've exceeded the maximum bucket size
            if self._use_static_shapes and len(decoder_input) > max_bucket:
                logger.warning(
                    f"Sequence length {len(decoder_input)} exceeds max bucket {max_bucket}. "
                    f"Stopping generation early."
                )
                break

            decoder_input_ids = np.array([decoder_input], dtype=np.int64)

            # Get logits
            logits = self._decode_step(encoder_hidden_states, decoder_input_ids, die_name)

            # Get next token (greedy decoding)
            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                next_token_logits = logits[0, :]
            else:
                break

            next_token = int(np.argmax(next_token_logits))

            # Check for end of transcript
            if next_token == self.EOT_TOKEN:
                break

            generated_tokens.append(next_token)
            decoder_input.append(next_token)

        # Decode to text
        text = self._decode_tokens(generated_tokens)
        return generated_tokens, text

    def _generate_with_kv_cache(
        self,
        encoder_hidden_states: np.ndarray,
        initial_tokens: List[int],
        die_name: str,
    ) -> Tuple[List[int], str]:
        """Generate with KV-cache (fast, incremental decoding).

        On first step, process all initial tokens and get KV-cache.
        On subsequent steps, only process the new token using cached keys/values.
        """
        generated_tokens = []
        past_key_values = None

        # First step: process all initial tokens
        decoder_input_ids = np.array([initial_tokens], dtype=np.int64)
        logits, past_key_values = self._decode_step_with_cache(
            encoder_hidden_states, decoder_input_ids, past_key_values, die_name
        )

        # Get first generated token
        if logits.ndim == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[0, :]

        next_token = int(np.argmax(next_token_logits))

        if next_token == self.EOT_TOKEN:
            text = self._decode_tokens(generated_tokens)
            return generated_tokens, text

        generated_tokens.append(next_token)

        # Subsequent steps: only process new token with cache
        for step in range(1, self.max_new_tokens):
            # Only pass the new token
            decoder_input_ids = np.array([[next_token]], dtype=np.int64)

            logits, past_key_values = self._decode_step_with_cache(
                encoder_hidden_states, decoder_input_ids, past_key_values, die_name
            )

            # Get next token
            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            else:
                next_token_logits = logits[0, :]

            next_token = int(np.argmax(next_token_logits))

            if next_token == self.EOT_TOKEN:
                break

            generated_tokens.append(next_token)

        text = self._decode_tokens(generated_tokens)
        return generated_tokens, text

    def _decode_step_with_cache(
        self,
        encoder_hidden_states: np.ndarray,
        decoder_input_ids: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]],
        die_name: str,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run decoder step with KV-cache support.

        Args:
            encoder_hidden_states: Encoder output
            decoder_input_ids: Current token(s) to process
            past_key_values: Previous KV-cache (None for first step)
            die_name: Device die to use for inference

        Returns:
            Tuple of (logits, new_past_key_values)
        """
        die_ctx = self._die_contexts[die_name]
        decoder_request = die_ctx["decoder"]

        inputs = {}

        # Set decoder input IDs
        if 'input_ids' in self._decoder_inputs:
            inputs[self._decoder_inputs['input_ids']] = decoder_input_ids
        else:
            inputs['decoder_input_ids'] = decoder_input_ids

        # Set encoder hidden states
        if 'encoder_hidden_states' in self._decoder_inputs:
            inputs[self._decoder_inputs['encoder_hidden_states']] = encoder_hidden_states
        else:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        # Set attention masks
        seq_len = decoder_input_ids.shape[1]
        if past_key_values is not None:
            # Add past sequence length
            past_len = list(past_key_values.values())[0].shape[2]
            total_len = past_len + seq_len
        else:
            total_len = seq_len

        if 'attention_mask' in self._decoder_inputs:
            attn_mask = np.ones((1, total_len), dtype=np.int64)
            inputs[self._decoder_inputs['attention_mask']] = attn_mask

        if 'encoder_attention_mask' in self._decoder_inputs:
            enc_seq_len = encoder_hidden_states.shape[1]
            enc_attn_mask = np.ones((1, enc_seq_len), dtype=np.int64)
            inputs[self._decoder_inputs['encoder_attention_mask']] = enc_attn_mask

        # Set past key values
        if past_key_values is not None:
            for name in self._kv_cache_inputs:
                if name in past_key_values:
                    inputs[name] = past_key_values[name]
        else:
            # Initialize with zeros for first step
            batch_size = 1
            for name in self._kv_cache_inputs:
                # Shape: (batch, num_heads, 0, head_dim) for empty cache
                inputs[name] = np.zeros((batch_size, self.NUM_ATTENTION_HEADS, 0, self.HEAD_DIM), dtype=np.float32)

        # Run inference
        decoder_request.infer(inputs)

        # Get logits
        logits = decoder_request.get_output_tensor(0).data.copy()

        # Get new past key values
        new_past = {}
        for name in self._kv_cache_outputs:
            tensor = decoder_request.get_tensor(name)
            new_past[name.replace('present', 'past_key_value')] = tensor.data.copy()

        return logits, new_past

    def _process_sample(self, sample_idx: int) -> str:
        """Process a single audio sample."""
        features = self.qsl.get_features(sample_idx)
        mel_features = features["input_features"]
        _, text = self._generate(mel_features)
        return text

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
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
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.2f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.2f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc=f"Whisper Offline ({self.device})")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response
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
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc=f"Whisper Server ({self.device})")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

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
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperOptimumNPUSUT:
    """
    Whisper SUT using Optimum-Intel with NPU device support.

    This implementation uses OVModelForSpeechSeq2Seq from Optimum-Intel
    which provides:
    - Automatic KV-cache handling
    - Efficient encoder-decoder inference
    - Proper token generation
    - Device placement via OpenVINO

    Recommended for NPU when Optimum-Intel is available.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: "LibriSpeechQSL",
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
        device: str = "NPU",
    ):
        """
        Initialize Whisper Optimum NPU SUT.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO Whisper model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
            device: OpenVINO device (NPU, CPU, etc.)
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "Optimum-Intel is required for WhisperOptimumNPUSUT. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens
        self.device = device

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model using Optimum-Intel with device specification."""
        from transformers import AutoProcessor

        logger.info(f"Loading Whisper model from {self.model_path} for {self.device}")

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Could not load processor from model path: {e}")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Configure OpenVINO for the target device
        ov_config = {
            "CACHE_DIR": "",
        }

        # Add device-specific properties
        if self.config.openvino.device_properties:
            ov_config.update(self.config.openvino.device_properties)

        # Load model with device specification
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            ov_config=ov_config,
            device=self.device,
            compile=True,
        )

        logger.info(f"Whisper model loaded on {self.device}")

    def _process_sample(self, sample_idx: int) -> str:
        """Process a single audio sample."""
        import torch

        features = self.qsl.get_features(sample_idx)
        input_features = features["input_features"]

        # Convert to tensor
        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)

        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        # Generate transcription
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=self.max_new_tokens,
            language="en",
            task="transcribe",
        )

        # Decode tokens to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
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
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.2f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.2f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc=f"Whisper Offline ({self.device})")

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
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc=f"Whisper Server ({self.device})")

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
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
