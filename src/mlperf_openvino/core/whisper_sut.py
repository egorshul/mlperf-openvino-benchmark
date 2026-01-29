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


class WhisperMultiDieSUT:
    """
    System Under Test for Whisper ASR on multi-die NPU.

    Uses OVModelForSpeechSeq2Seq from optimum-intel for correct generation.
    Distributes inference across multiple NPU dies (or falls back to CPU).
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
            encoder_path: Path to encoder OpenVINO model (used to find model dir)
            decoder_path: Path to decoder OpenVINO model
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "Optimum-Intel is required. Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(encoder_path).parent  # Model directory
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

        # Multi-die models: list of (die_name, model) tuples
        self._models: List[Tuple[str, Any]] = []
        self._model_index = 0

        # Processor (shared)
        self.processor = None

        # Setup models
        self._setup_models()

    def _setup_models(self) -> None:
        """Setup OVModelForSpeechSeq2Seq on CPU.

        Note: NPU compilation of Whisper decoder fails due to unsupported ops.
        Using CPU with optimum-intel ensures correct generation and accuracy.
        """
        from transformers import AutoProcessor

        logger.info("Setting up Whisper model on CPU (decoder not supported on NPU)")

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            logger.info("Loading processor from openai/whisper-large-v3")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Load model on CPU
        logger.info("  Loading model...")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            device="CPU",
            ov_config={"CACHE_DIR": ""},
            compile=True,
        )
        self._models.append(("CPU", model))

        logger.info("Whisper SUT ready (CPU)")

    def _get_next_model(self) -> Tuple[str, Any]:
        """Get next model in round-robin fashion."""
        die_name, model = self._models[self._model_index]
        self._model_index = (self._model_index + 1) % len(self._models)
        return die_name, model

    def _process_sample(self, sample_idx: int, model: Any) -> str:
        """Process a single sample using given model."""
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

        # Generate transcription using model.generate() - handles KV-cache automatically
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
        self._die_index = 0
