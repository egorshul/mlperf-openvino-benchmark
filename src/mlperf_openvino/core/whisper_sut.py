"""
Whisper-specific System Under Test implementation.

This module provides SUT implementation optimized for Whisper ASR model,
handling encoder-decoder architecture and tokenization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, Scenario
from ..backends.base import BaseBackend
from ..datasets.librispeech import LibriSpeechQSL

logger = logging.getLogger(__name__)


class WhisperSUT:
    """
    System Under Test for Whisper ASR model.
    
    Whisper has an encoder-decoder architecture:
    - Encoder: Processes mel spectrogram to hidden states
    - Decoder: Generates text tokens autoregressively
    
    For MLPerf, we typically benchmark the full encode-decode pipeline.
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
        max_new_tokens: int = 448,
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
        
        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0
        
        # Create LoadGen handles
        self._sut = None
        self._qsl = None
        
        # Tokenizer for decoding (lazy loaded)
        self._tokenizer = None
    
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
        inputs = {
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_input_ids": decoder_input_ids,
        }
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
        
        for _ in range(self.max_new_tokens):
            # Prepare decoder input
            decoder_input_ids = np.array([decoder_input], dtype=np.int64)
            
            # Get logits
            logits = self._decode_step(encoder_hidden_states, decoder_input_ids)
            
            # Get next token (greedy or sampling)
            next_token_logits = logits[0, -1, :]
            
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
        
        _, text = self._generate(mel_features)
        
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
        
        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1
            
            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            
            # Create response (using dummy data for LoadGen)
            response_array = np.array([len(text)], dtype=np.int64)
            response = lg.QuerySampleResponse(
                sample.id,
                response_array.ctypes.data,
                response_array.nbytes
            )
            responses.append(response)
        
        lg.QuerySamplesComplete(responses)
    
    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1
            
            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text
            
            # Create response
            response_array = np.array([len(text)], dtype=np.int64)
            response = lg.QuerySampleResponse(
                sample.id,
                response_array.ctypes.data,
                response_array.nbytes
            )
            lg.QuerySamplesComplete([response])
    
    def get_sut(self):
        """Get LoadGen SUT handle."""
        if self._sut is None:
            self._sut = lg.ConstructSUT(
                self.issue_queries,
                lambda: None  # flush_queries
            )
        return self._sut
    
    def get_qsl(self):
        """Get LoadGen QSL handle."""
        if self._qsl is None:
            self._qsl = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl
    
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
        
        self._sut = None
        self._qsl = None
    
    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)
        
        responses = []
        
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
            
            # Create response
            response = lg.QuerySampleResponse(
                sample.id,
                encoder_output.ctypes.data,
                encoder_output.nbytes
            )
            responses.append(response)
        
        lg.QuerySamplesComplete(responses)
    
    def get_sut(self):
        """Get LoadGen SUT handle."""
        if self._sut is None:
            self._sut = lg.ConstructSUT(
                self.issue_queries,
                lambda: None
            )
        return self._sut
    
    def get_qsl(self):
        """Get LoadGen QSL handle."""
        if self._qsl is None:
            self._qsl = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl
    
    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all encoder outputs."""
        return self._predictions.copy()
    
    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
