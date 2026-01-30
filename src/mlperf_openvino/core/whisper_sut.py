"""
Whisper-specific System Under Test implementation.

This module provides SUT implementation optimized for Whisper ASR model,
using optimum-intel OVModelForSpeechSeq2Seq for proper encoder-decoder inference.
"""

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


class WhisperHybridEncoder:
    """
    OpenVINO Encoder wrapper for hybrid NPU+CPU inference.

    Based on optimum-intel OVEncoder, but allows separate device compilation.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "NPU",
        ov_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize encoder.

        Args:
            model_path: Path to encoder_model.xml
            device: Device to compile on (NPU, CPU)
            ov_config: OpenVINO configuration
        """
        import openvino as ov

        self.device = device
        self.ov_config = ov_config or {}

        # Load model
        core = ov.Core()
        logger.info(f"  Loading encoder model from {model_path}")
        self.model = core.read_model(str(model_path))

        # Get input/output names
        self.input_names = [inp.get_any_name() for inp in self.model.inputs]
        self.output_names = [out.get_any_name() for out in self.model.outputs]
        logger.info(f"  Encoder inputs: {self.input_names}")
        logger.info(f"  Encoder outputs: {self.output_names}")

        # Find main input name
        self.main_input_name = "input_features"
        for name in self.input_names:
            if "input_features" in name or "input" in name.lower():
                self.main_input_name = name
                break

        # Check device availability
        available_devices = core.available_devices
        logger.info(f"  Available devices: {available_devices}")
        if device not in available_devices and not any(d.startswith(device) for d in available_devices):
            raise RuntimeError(f"Device {device} not available. Available: {available_devices}")

        # Reshape to static shape for accelerators (NPU, etc.)
        # Whisper encoder expects (batch=1, n_mels=128, time=3000)
        if device != "CPU":
            logger.info(f"  Checking encoder shape for {device}...")
            try:
                # Get current input shape
                input_shape = self.model.input(0).get_partial_shape()
                logger.info(f"  Current input shape: {input_shape}")

                # Check if reshape is needed (only if shape is dynamic)
                if input_shape.is_dynamic:
                    # Whisper Large V3: batch=1, n_mels=128, time=3000
                    static_shape = [1, 128, 3000]
                    self.model.reshape({self.main_input_name: static_shape})
                    logger.info(f"  Reshaped to static: {static_shape}")
                else:
                    logger.info(f"  Shape is already static, no reshape needed")
            except Exception as e:
                logger.warning(f"  Could not check/reshape encoder: {e}")

        # Compile model - use provided config for accelerators, minimal for CPU
        logger.info(f"Compiling encoder on {device}...")
        if device == "CPU":
            # Use minimal config for CPU
            compile_config = {"CACHE_DIR": ""}
        else:
            # Use user-provided config (-p options) for accelerators
            compile_config = self.ov_config
            logger.info(f"  Using config for accelerator: {compile_config}")

        try:
            self.compiled_model = core.compile_model(self.model, device, compile_config)
            self.request = self.compiled_model.create_infer_request()
            logger.info(f"Encoder compiled on {device}")
        except Exception as e:
            logger.error(f"Failed to compile encoder on {device}: {e}")
            raise

    def __call__(self, input_features, **kwargs):
        """
        Run encoder inference.

        Matches optimum-intel OVEncoder.forward() exactly:
        - Uses start_async(share_inputs=True) + wait() instead of infer()
        - Passes input tensor directly (torch or numpy)
        - Gets output by name, clones to own the data

        Args:
            input_features: Mel spectrogram tensor (batch, n_mels, time)

        Returns:
            BaseModelOutput with last_hidden_state
        """
        import torch
        from transformers.modeling_outputs import BaseModelOutput

        # Pass input directly (match optimum-intel: no numpy conversion)
        inputs = {self.main_input_name: input_features}

        # Run inference (match optimum-intel: start_async + wait with share_inputs)
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        # Get output by name and clone (match optimum-intel exactly)
        last_hidden_state_data = self.request.get_tensor("last_hidden_state").data

        # Log encoder output stats for first call (debugging)
        if not hasattr(self, '_encoder_logged'):
            self._encoder_logged = True
            logger.info(f"Encoder output stats ({self.device}):")
            logger.info(f"  Shape: {last_hidden_state_data.shape}")
            logger.info(f"  Dtype: {last_hidden_state_data.dtype}")
            logger.info(f"  Min: {last_hidden_state_data.min():.6f}, Max: {last_hidden_state_data.max():.6f}")
            logger.info(f"  Mean: {last_hidden_state_data.mean():.6f}, Std: {last_hidden_state_data.std():.6f}")
            logger.info(f"  First values [0,0,:5]: {last_hidden_state_data[0, 0, :5]}")
            logger.info(f"  First values [0,:5,0]: {last_hidden_state_data[0, :5, 0]}")

        # Clone to own the data (match optimum-intel: .clone())
        last_hidden_state = torch.from_numpy(last_hidden_state_data).clone()

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class WhisperHybridDecoder:
    """
    OpenVINO Decoder wrapper with stateful KV-cache support.

    Based on optimum-intel OVDecoder, but allows separate device compilation.
    Supports both stateful (with internal KV-cache) and stateless models.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize decoder.

        Args:
            model_path: Path to decoder_model.xml or decoder_with_past_model.xml
            device: Device to compile on (CPU recommended for decoder)
            ov_config: OpenVINO configuration
        """
        import openvino as ov

        self.device = device
        self.ov_config = ov_config or {}

        # Load model
        core = ov.Core()
        self.model = core.read_model(str(model_path))

        # Get input/output names
        self.input_names = [inp.get_any_name() for inp in self.model.inputs]
        self.output_names = [out.get_any_name() for out in self.model.outputs]

        # Detect KV-cache inputs/outputs
        self.key_value_input_names = [k for k in self.input_names if "key_values" in k or "past_key" in k]
        self.key_value_output_names = [k for k in self.output_names if "key_values" in k or "present" in k]

        # Check if model is stateful (has internal state for KV-cache)
        self.stateful = any(self.model.get_variable_state(state.name) is not None
                          for state in self.model.outputs if hasattr(state, 'name'))
        try:
            # Actually check for state variables
            self.stateful = len(self.model.get_variables()) > 0
        except Exception:
            self.stateful = False

        self.use_past = len(self.key_value_input_names) > 0 or self.stateful
        self.next_beam_idx = None
        self._past_length = 0

        # Compile model
        logger.info(f"Compiling decoder on {device}...")
        self.compiled_model = core.compile_model(self.model, device, self.ov_config)
        self.request = self.compiled_model.create_infer_request()

        logger.info(f"Decoder compiled on {device} (stateful={self.stateful}, use_past={self.use_past})")
        logger.info(f"  Decoder input names: {self.input_names}")
        logger.info(f"  Decoder output names: {self.output_names}")
        # Log expected input shapes (critical for diagnosing prefill vs one-at-a-time)
        for inp in self.model.inputs:
            name = inp.get_any_name()
            pshape = inp.get_partial_shape()
            logger.info(f"  Input '{name}': shape={pshape}, type={inp.get_element_type()}")
        for out in self.model.outputs:
            name = out.get_any_name()
            pshape = out.get_partial_shape()
            logger.info(f"  Output '{name}': shape={pshape}, type={out.get_element_type()}")
        if self.key_value_input_names:
            logger.info(f"  KV-cache inputs: {self.key_value_input_names}")
        if self.key_value_output_names:
            logger.info(f"  KV-cache outputs: {self.key_value_output_names}")

    def __call__(
        self,
        input_ids,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Run decoder inference step.

        Matches optimum-intel OVDecoder.forward() exactly:
        - Passes torch tensors directly (no numpy conversion)
        - Same input ordering and conditional logic
        - Same start_async(share_inputs=True) pattern
        """
        import torch
        from transformers.modeling_outputs import Seq2SeqLMOutput

        # Model inputs dict — match optimum-intel OVDecoder.forward() exactly
        inputs = {}

        # Reset state for first token (match optimum-intel)
        if self.stateful and past_key_values is None:
            self.request.reset_state()
            self._past_length = 0
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=np.int32)

        # Add past_key_values for non-stateful models (match optimum-intel order: before input_ids)
        if past_key_values is not None and not self.stateful:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Pass input_ids directly as torch tensor (match optimum-intel: no numpy conversion)
        inputs["input_ids"] = input_ids

        # Add attention mask (match optimum-intel conditional)
        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        # Add encoder attention mask (match optimum-intel conditional)
        if "encoder_attention_mask" in self.input_names and encoder_attention_mask is not None:
            inputs["encoder_attention_mask"] = encoder_attention_mask

        # Add encoder hidden states (match optimum-intel conditional)
        if "encoder_hidden_states" in self.input_names and encoder_hidden_states is not None:
            inputs["encoder_hidden_states"] = encoder_hidden_states

        # Add cache position (match optimum-intel)
        if "cache_position" in self.input_names:
            if cache_position is None:
                past_len = self._past_length if self.stateful else 0
                cache_position = np.arange(past_len, past_len + input_ids.shape[1])
            inputs["cache_position"] = cache_position

        # Add beam_idx (match optimum-intel)
        if "beam_idx" in self.input_names:
            batch_size = input_ids.shape[0]
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None
                else np.arange(batch_size, dtype=np.int32)
            )

        # Run inference (match optimum-intel: start_async + wait with share_inputs)
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        # Get logits — clone to own the data (match optimum-intel exactly)
        logits = torch.from_numpy(self.request.get_tensor("logits").data).clone()
        self._past_length += input_ids.shape[1]

        # Get output past_key_values (for non-stateful models)
        out_past_key_values = ((),)
        if not self.stateful and self.key_value_output_names:
            out_past = tuple(
                np.copy(self.request.get_tensor(key).data)
                for key in self.key_value_output_names
            )
            num_per_layer = 2
            out_past_key_values = tuple(
                out_past[i:i + num_per_layer]
                for i in range(0, len(out_past), num_per_layer)
            )

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        if self.stateful:
            # Use int32 to match model's expected beam_idx type (int32_t)
            self.next_beam_idx = np.array(beam_idx, dtype=np.int32)
            return past_key_values
        else:
            reordered = ()
            for layer_past in past_key_values:
                reordered += (
                    tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past[:2])
                    + layer_past[2:],
                )
            return reordered

    def _get_past_length(self, past_key_values=None):
        """Get length of past cache."""
        if self.stateful:
            return self._past_length
        if past_key_values is None or len(past_key_values) == 0:
            return 0
        return past_key_values[0][0].shape[-2]


class WhisperHybridModel:
    """
    Whisper model with hybrid NPU+CPU execution.

    Encoder runs on NPU for fast feature processing.
    Decoder runs on CPU (NPU doesn't support all decoder ops).
    Uses WhisperForConditionalGeneration.generate() for proper generation.
    """

    # Required attributes for WhisperForConditionalGeneration.generate()
    main_input_name = "input_features"

    def __init__(
        self,
        encoder: WhisperHybridEncoder,
        decoder: WhisperHybridDecoder,
        config,
        generation_config=None,
    ):
        """
        Initialize hybrid model.

        Args:
            encoder: WhisperHybridEncoder instance (on NPU)
            decoder: WhisperHybridDecoder instance (on CPU)
            config: WhisperConfig
            generation_config: GenerationConfig
        """
        from transformers import GenerationConfig

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.generation_config = generation_config or GenerationConfig.from_model_config(config)

        # Required attributes for generate()
        self.device = "cpu"  # Outputs on CPU

        # Flag for one-time generation info logging
        self._generation_logged = False

        # Dummy model attribute for Whisper generate() stride calculation
        class DummyEncoder:
            class Conv:
                def __init__(self, stride):
                    self.stride = stride
            conv1 = Conv(stride=(1,))
            conv2 = Conv(stride=(2,))

        class DummyModel:
            encoder = DummyEncoder()

        self.model = DummyModel()

    def get_encoder(self):
        """Return encoder for generate()."""
        return self.encoder

    def get_decoder(self):
        """Return decoder for generate()."""
        return self.decoder

    def can_generate(self):
        """Model can generate text."""
        return True

    def forward(
        self,
        input_features=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Forward pass for generation.

        Args:
            input_features: Mel spectrogram (optional if encoder_outputs provided)
            decoder_input_ids: Current decoder tokens
            encoder_outputs: Pre-computed encoder outputs
            past_key_values: KV-cache
            attention_mask: Encoder attention mask
            decoder_attention_mask: Decoder attention mask
            cache_position: Position in cache

        Returns:
            Seq2SeqLMOutput with logits
        """
        # Run encoder if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features)

        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Run decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        return decoder_outputs

    def __call__(self, *args, **kwargs):
        """Make model callable."""
        return self.forward(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        """Prepare inputs for generation step (from _OVModelForWhisper)."""
        import torch

        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            past_length = self.decoder._get_past_length(past_key_values)
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
            if decoder_position_ids is not None:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]
                decoder_position_ids = decoder_position_ids.clone(memory_format=torch.contiguous_format)

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1]:]

        decoder_input_ids = decoder_input_ids.contiguous()

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    def generate(
        self,
        input_features,
        max_new_tokens: int = 440,
        language: str = "en",
        task: str = "transcribe",
        **kwargs,
    ):
        """
        Generate transcription tokens from mel spectrogram.

        Matches HF WhisperForConditionalGeneration.generate() behavior:
        - Feeds forced tokens ONE AT A TIME (not all-at-once prefill)
        - Uses suppress_tokens and begin_suppress_tokens from generation_config
        - Greedy decoding (temperature=0)

        Args:
            input_features: Mel spectrogram tensor (batch, n_mels, time)
            max_new_tokens: Maximum tokens to generate
            language: Language code
            task: Task type ("transcribe" or "translate")

        Returns:
            Generated token IDs tensor
        """
        import torch

        # Load ALL special tokens from generation_config (model-specific).
        # Whisper v2 and v3 have DIFFERENT token IDs — must not hardcode!
        gc = self.generation_config

        SOT_TOKEN = getattr(gc, 'decoder_start_token_id', 50258)
        EOT_TOKEN = getattr(gc, 'eos_token_id', 50257)
        NO_TIMESTAMPS_TOKEN = getattr(gc, 'no_timestamps_token_id', 50364)

        # Task tokens from generation_config.task_to_id
        task_to_id = getattr(gc, 'task_to_id', {})
        TRANSCRIBE_TOKEN = task_to_id.get("transcribe", 50360)
        TRANSLATE_TOKEN = task_to_id.get("translate", 50359)

        # Language tokens from generation_config.lang_to_id
        lang_to_id = getattr(gc, 'lang_to_id', {})
        LANGUAGE_TOKENS = {}
        for lang_tag, token_id in lang_to_id.items():
            # "<|en|>" -> "en"
            lang_code = lang_tag.strip("<|>")
            LANGUAGE_TOKENS[lang_code] = token_id
        if not LANGUAGE_TOKENS:
            # Fallback for v3 if lang_to_id missing
            LANGUAGE_TOKENS = {"en": 50259}

        SUPPRESS_TOKENS = getattr(gc, 'suppress_tokens', None) or []
        BEGIN_SUPPRESS_TOKENS = getattr(gc, 'begin_suppress_tokens', None) or [220, 50257]

        # Diagnostic: first sample flag
        is_first_sample = not self._generation_logged

        # Log generation config once
        if is_first_sample:
            logger.info("=" * 60)
            logger.info("Whisper Generation Configuration:")
            logger.info(f"  Encoder device: {self.encoder.device}")
            logger.info(f"  Decoder device: {self.decoder.device}")
            logger.info(f"  Decoder stateful: {self.decoder.stateful}")
            logger.info(f"  Language: {language} (token {LANGUAGE_TOKENS.get(language, 'unknown')})")
            logger.info(f"  Task: {task}")
            logger.info(f"  Max new tokens: {max_new_tokens}")
            logger.info(f"  Decoding: greedy (temperature=0)")
            logger.info(f"  suppress_tokens: {len(SUPPRESS_TOKENS)} tokens (from generation_config)")
            logger.info(f"  begin_suppress_tokens: {BEGIN_SUPPRESS_TOKENS}")
            logger.info(f"  Token IDs: SOT={SOT_TOKEN}, EOT={EOT_TOKEN}, "
                       f"transcribe={TRANSCRIBE_TOKEN}, translate={TRANSLATE_TOKEN}, "
                       f"no_timestamps={NO_TIMESTAMPS_TOKEN}")
            task_token_val = TRANSCRIBE_TOKEN if task == 'transcribe' else TRANSLATE_TOKEN
            lang_token_val = LANGUAGE_TOKENS.get(language, '?')
            logger.info(f"  Forced init tokens: [{SOT_TOKEN}, {lang_token_val}, "
                       f"{task_token_val}, {NO_TIMESTAMPS_TOKEN}]")
            logger.info(f"  Feeding: ONE token at a time (matches HF generate)")
            logger.info("=" * 60)
            self._generation_logged = True

        batch_size = input_features.shape[0]
        device = input_features.device if hasattr(input_features, 'device') else 'cpu'

        # Run encoder
        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Create encoder_attention_mask if required by decoder
        encoder_attention_mask = None
        if "encoder_attention_mask" in self.decoder.input_names:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len),
                dtype=torch.long,
                device=device,
            )

        # Build forced token sequence: [SOT, language, task, no_timestamps]
        task_token = TRANSCRIBE_TOKEN if task == "transcribe" else TRANSLATE_TOKEN
        language_token = LANGUAGE_TOKENS.get(language, LANGUAGE_TOKENS["en"])
        forced_tokens = [SOT_TOKEN, language_token, task_token, NO_TIMESTAMPS_TOKEN]

        # === Feed forced tokens ONE AT A TIME ===
        # This matches HF WhisperForConditionalGeneration.generate() which uses
        # prepare_inputs_for_generation() to strip prefix and feed one token per step.
        # The stateful decoder accumulates KV-cache internally.
        generated_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        past_key_values = None

        for i, token_id in enumerate(forced_tokens):
            input_token = torch.tensor(
                [[token_id]], dtype=torch.long, device=device
            ).expand(batch_size, -1)

            # First token: reset state; subsequent tokens: continue state
            if i == 0:
                pkv = None  # → reset_state() in decoder
            else:
                pkv = ((),) if self.decoder.stateful else past_key_values

            outputs = self.decoder(
                input_ids=input_token,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=pkv,
            )

            if not self.decoder.stateful:
                past_key_values = outputs.past_key_values

            generated_ids = torch.cat([generated_ids, input_token], dim=-1)

            if is_first_sample:
                logits_diag = outputs.logits
                top_logits = logits_diag[0, -1, :] if logits_diag.dim() == 3 else logits_diag[0, :]
                top5_vals, top5_ids = torch.topk(top_logits, 5)
                logger.info(f"  Forced[{i}] input={token_id}, logits top5: "
                           f"{list(zip(top5_ids.tolist(), [f'{v:.2f}' for v in top5_vals.tolist()]))}")

        # === First free generation step ===
        # Get logits from last forced token's output
        logits = outputs.logits
        next_token_logits = logits[:, -1, :].clone() if logits.dim() == 3 else logits.clone()

        vocab_size = next_token_logits.shape[-1]

        # Apply suppress_tokens (always) + begin_suppress_tokens (first free position only)
        for token_id in SUPPRESS_TOKENS:
            if token_id < vocab_size:
                next_token_logits[:, token_id] = float('-inf')
        for token_id in BEGIN_SUPPRESS_TOKENS:
            if token_id < vocab_size:
                next_token_logits[:, token_id] = float('-inf')

        next_tokens = next_token_logits.argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

        if is_first_sample:
            top5_vals, top5_ids = torch.topk(next_token_logits[0], 5)
            logger.info(f"  Free[0] (after suppress): token={next_tokens[0].item()}, top5: "
                       f"{list(zip(top5_ids.tolist(), [f'{v:.2f}' for v in top5_vals.tolist()]))}")

        if (next_tokens == EOT_TOKEN).all():
            if is_first_sample:
                logger.info(f"  EOT at step 0, total tokens: {generated_ids.shape[1]}")
            return generated_ids

        # === Auto-regressive generation: one token at a time ===
        for step in range(1, max_new_tokens):
            input_ids = generated_ids[:, -1:]

            pkv = ((),) if self.decoder.stateful else past_key_values
            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=pkv,
            )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :].clone() if logits.dim() == 3 else logits.clone()

            # Apply suppress_tokens (not begin_suppress_tokens — only at first position)
            for token_id in SUPPRESS_TOKENS:
                if token_id < vocab_size:
                    next_token_logits[:, token_id] = float('-inf')

            next_tokens = next_token_logits.argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if not self.decoder.stateful:
                past_key_values = outputs.past_key_values

            # Diagnostic: log first 10 generated tokens
            if is_first_sample and step <= 10:
                top5_vals, top5_ids = torch.topk(next_token_logits[0], 5)
                logger.info(f"  Free[{step}]: token={next_tokens[0].item()}, top5: "
                           f"{list(zip(top5_ids.tolist(), [f'{v:.2f}' for v in top5_vals.tolist()]))}")

            if (next_tokens == EOT_TOKEN).all():
                if is_first_sample:
                    logger.info(f"  EOT at step {step}, total tokens: {generated_ids.shape[1]}")
                break

        if is_first_sample:
            logger.info(f"  Generation done: {generated_ids.shape[1]} tokens "
                       f"(forced={len(forced_tokens)}, generated={generated_ids.shape[1] - len(forced_tokens)})")
            logger.info(f"  First 20 token IDs: {generated_ids[0, :20].tolist()}")

        return generated_ids


class WhisperMultiDieSUT:
    """
    System Under Test for Whisper ASR on multi-die NPU.

    Uses OVModelForSpeechSeq2Seq from optimum-intel for correct generation,
    with encoder re-compiled on accelerator dies for hardware acceleration.
    This guarantees the same accuracy as the CPU reference (~98%).
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
        """
        Initialize Whisper Multi-Die SUT.

        Args:
            config: Benchmark configuration
            encoder_path: Path to encoder OpenVINO model
            decoder_path: Path to decoder OpenVINO model
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.model_path = self.encoder_path.parent  # Model directory
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
        self._progress_update_interval = 0.5

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Multi-die models: list of (die_name, OVModelForSpeechSeq2Seq) tuples
        self._models: List[Tuple[str, Any]] = []
        self._model_index = 0

        # Processor (shared)
        self.processor = None

        # Setup models
        self._setup_models()

    def _discover_device_dies(self, device: str) -> List[str]:
        """Discover available dies for the specified device.

        Args:
            device: Base device name (e.g., "NPU", "GPU", "X")

        Returns:
            List of device dies with numbers only (e.g., ["X.0", "X.1"])
            Returns empty list if no numbered dies found.
        """
        import openvino as ov
        import re

        core = ov.Core()
        devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")

        # Find dies for this device (format: DEVICE.N where N is a number)
        # Only return dies with numeric suffix (X.0, X.1, etc.)
        pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
        device_dies = [d for d in devices if pattern.match(d)]
        logger.info(f"Matching dies for '{device}': {device_dies}")

        return sorted(device_dies)

    def _setup_models(self) -> None:
        """Setup models using OVModelForSpeechSeq2Seq with encoder on accelerator.

        Loads the proven-correct optimum-intel model on CPU, then re-compiles
        the encoder on each accelerator die for hardware acceleration.
        This uses HF's WhisperForConditionalGeneration.generate() internally,
        which correctly handles forced_decoder_ids, suppress_tokens, KV-cache, etc.
        """
        import os
        import re
        from transformers import AutoProcessor

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum-intel is required for WhisperMultiDieSUT. "
                "Install with: pip install optimum[openvino]"
            )

        # Get target device from config
        target_device = self.config.openvino.device if hasattr(self.config, 'openvino') else "CPU"

        # Allow overriding encoder device via environment variable for diagnostics.
        # Set WHISPER_ENCODER_DEVICE=CPU to run encoder on CPU (isolates encoder vs generate bugs).
        encoder_device_override = os.environ.get("WHISPER_ENCODER_DEVICE", "").strip()
        if encoder_device_override:
            logger.info(f"WHISPER_ENCODER_DEVICE override: {encoder_device_override} (was {target_device})")
            target_device = encoder_device_override

        logger.info(f"Setting up Whisper Multi-Die SUT (encoder={target_device}, decoder=CPU)")

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            logger.info("Loading processor from openai/whisper-large-v3")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Determine which devices to use for encoders
        if target_device == "CPU":
            device_dies = ["CPU"]
            logger.info("Using CPU encoder (no die discovery)")
        elif re.match(r"^.+\.\d+$", target_device):
            device_dies = [target_device]
            logger.info(f"Using specified device die: {target_device}")
        else:
            device_dies = self._discover_device_dies(target_device)
            if device_dies:
                logger.info(f"Discovered {len(device_dies)} device dies: {device_dies}")
            else:
                logger.warning(f"No {target_device} dies found, falling back to CPU")
                device_dies = ["CPU"]

        # Create one OVModelForSpeechSeq2Seq per die.
        # Uses optimum-intel's proven generate() (same code as CPU SUT ~98% accuracy).
        for die in device_dies:
            logger.info(f"Loading OVModelForSpeechSeq2Seq for {die}...")
            try:
                model = self._load_optimum_model()

                # If die is not CPU, re-compile encoder on accelerator
                if die != "CPU":
                    self._patch_encoder_device(model, die)

                self._models.append((die, model))
                logger.info(f"  Model ready on {die}")
            except Exception as e:
                logger.warning(f"Failed to create model for {die}: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to pure CPU if no models created
        if not self._models:
            logger.info("Creating CPU fallback model...")
            model = self._load_optimum_model()
            self._models.append(("CPU", model))

        # Log summary
        encoder_devices = [die for die, _ in self._models]
        logger.info("=" * 60)
        logger.info("Whisper Multi-Die SUT Configuration:")
        logger.info(f"  Backend: OVModelForSpeechSeq2Seq (optimum-intel)")
        logger.info(f"  Encoder device(s): {encoder_devices}")
        logger.info(f"  Decoder device: CPU")
        logger.info(f"  Max new tokens: {self.max_new_tokens}")
        logger.info(f"  Scenario: {self.scenario}")
        logger.info(f"  Total models: {len(self._models)}")
        logger.info("=" * 60)

    def _load_optimum_model(self):
        """Load OVModelForSpeechSeq2Seq on CPU.

        Tries default optimum-intel file names first, falls back to
        detecting file names from encoder_path/decoder_path.

        Returns:
            OVModelForSpeechSeq2Seq model instance
        """
        model_dir = str(self.model_path)

        # Try default file names first (openvino_ prefix from optimum-cli)
        try:
            return OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                device="CPU",
                ov_config={"CACHE_DIR": ""},
            )
        except Exception as e:
            logger.info(f"Default file names failed ({e}), trying detected names...")

        # Fall back: use file names we know exist (from benchmark_runner discovery)
        enc_name = self.encoder_path.name
        dec_name = self.decoder_path.name
        logger.info(f"  Using encoder={enc_name}, decoder={dec_name}")

        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            device="CPU",
            ov_config={"CACHE_DIR": ""},
            encoder_file_name=enc_name,
            decoder_file_name=dec_name,
        )

    def _patch_encoder_device(self, model, die: str) -> None:
        """Re-compile model's encoder on the specified accelerator device.

        Reads the encoder model fresh from disk, reshapes to static shape
        (required by accelerators), compiles on the target die, and replaces
        the model's encoder request.

        Args:
            model: OVModelForSpeechSeq2Seq instance (loaded on CPU)
            die: Target device (e.g., "X.0", "NPU.0")
        """
        import openvino as ov

        core = ov.Core()

        # Read encoder model fresh from disk (need own copy for static reshape)
        encoder_model = core.read_model(str(self.encoder_path))

        # Reshape to static shape for accelerator (NPU requires static shapes)
        # Whisper Large V3: batch=1, n_mels=128, time=3000
        for inp in encoder_model.inputs:
            if inp.get_partial_shape().is_dynamic:
                input_name = inp.get_any_name()
                static_shape = [1, 128, 3000]
                encoder_model.reshape({input_name: static_shape})
                logger.info(f"  Encoder reshaped to {static_shape} for {die}")
                break

        # Build accelerator config from user's -p options
        ov_config = {"CACHE_DIR": ""}
        if hasattr(self.config, 'openvino') and hasattr(self.config.openvino, 'device_properties'):
            for key, value in self.config.openvino.device_properties.items():
                ov_config[key] = value

        # Compile encoder on accelerator die
        logger.info(f"  Compiling encoder on {die} with config: {ov_config}")
        compiled_encoder = core.compile_model(encoder_model, die, ov_config)

        # Replace encoder's request with accelerator-compiled version.
        # OVEncoder.forward() calls self.request(inputs, share_inputs=True, share_outputs=True)
        # where self.request is a CompiledModel — our compiled_encoder is also a CompiledModel.
        model.encoder.request = compiled_encoder
        logger.info(f"  Encoder patched to {die}")

    def _get_next_model(self) -> Tuple[str, Any]:
        """Get next model in round-robin fashion."""
        die_name, model = self._models[self._model_index]
        self._model_index = (self._model_index + 1) % len(self._models)
        return die_name, model

    def _process_sample(self, sample_idx: int, model: Any) -> str:
        """Process a single sample using OVModelForSpeechSeq2Seq.generate()."""
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

        # Generate transcription using optimum-intel's proven generate()
        # (WhisperForConditionalGeneration.generate → handles forced_decoder_ids,
        #  suppress_tokens, KV-cache, etc. correctly)
        generated_ids = model.generate(
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
        """Update progress."""
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

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_queries_offline(query_samples)
        else:
            self._issue_queries_server(query_samples)

    def _issue_queries_offline(self, query_samples: List[Any]) -> None:
        """Process all samples in Offline mode."""
        total = len(query_samples)
        self._start_progress(total, "Whisper Multi-Die Offline")

        responses = []
        response_arrays = []

        for sample in query_samples:
            sample_idx = sample.index

            # Select model (round-robin across dies)
            die_name, model = self._get_next_model()

            # Generate transcription using model.generate()
            text = self._process_sample(sample_idx, model)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            # Create response
            text_bytes = text.encode('utf-8')
            response_array = array.array('B', text_bytes)
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_queries_server(self, query_samples: List[Any]) -> None:
        """Process samples in Server mode (one at a time)."""
        for sample in query_samples:
            sample_idx = sample.index

            # Select model (round-robin across dies)
            die_name, model = self._get_next_model()

            # Generate transcription using model.generate()
            text = self._process_sample(sample_idx, model)
            self._predictions[sample_idx] = text
            self._sample_count += 1

            # Respond immediately
            text_bytes = text.encode('utf-8')
            response_array = array.array('B', text_bytes)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self) -> None:
        """Flush pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries
            )
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
        self._model_index = 0
