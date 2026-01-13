"""
Tests for LibriSpeech dataset and Whisper model support.
"""

import pytest
import numpy as np
from pathlib import Path


class TestLibriSpeechConstants:
    """Tests for LibriSpeech constants."""
    
    def test_dataset_import(self):
        """Test that dataset can be imported."""
        from mlperf_openvino.datasets.librispeech import (
            LibriSpeechDataset,
            LibriSpeechQSL,
            SAMPLE_RATE,
            N_MELS,
        )
        
        assert SAMPLE_RATE == 16000
        assert N_MELS == 80
    
    def test_dataset_constants(self):
        """Test audio processing constants."""
        from mlperf_openvino.datasets.librispeech import (
            SAMPLE_RATE,
            N_FFT,
            HOP_LENGTH,
            N_MELS,
            CHUNK_LENGTH,
            N_SAMPLES,
        )
        
        assert SAMPLE_RATE == 16000
        assert N_FFT == 400
        assert HOP_LENGTH == 160
        assert N_MELS == 80
        assert CHUNK_LENGTH == 30
        assert N_SAMPLES == CHUNK_LENGTH * SAMPLE_RATE


class TestPadOrTrim:
    """Tests for pad_or_trim function."""
    
    def test_padding(self):
        """Test padding short audio."""
        from mlperf_openvino.datasets.librispeech import pad_or_trim
        
        short_audio = np.zeros(1000)
        padded = pad_or_trim(short_audio, 2000)
        assert len(padded) == 2000
    
    def test_trimming(self):
        """Test trimming long audio."""
        from mlperf_openvino.datasets.librispeech import pad_or_trim
        
        long_audio = np.zeros(5000)
        trimmed = pad_or_trim(long_audio, 2000)
        assert len(trimmed) == 2000
    
    def test_exact_length(self):
        """Test exact length audio."""
        from mlperf_openvino.datasets.librispeech import pad_or_trim
        
        exact_audio = np.zeros(2000)
        result = pad_or_trim(exact_audio, 2000)
        assert len(result) == 2000


class TestLibriSpeechQSL:
    """Tests for LibriSpeech Query Sample Library."""
    
    def test_qsl_creation(self):
        """Test QSL creation."""
        from mlperf_openvino.datasets.librispeech import LibriSpeechQSL
        
        qsl = LibriSpeechQSL(
            data_path="/tmp/dummy",
            performance_sample_count=100,
        )
        
        assert qsl._performance_sample_count == 100
    
    def test_qsl_performance_sample_count_default(self):
        """Test default performance sample count."""
        from mlperf_openvino.datasets.librispeech import LibriSpeechQSL
        
        qsl = LibriSpeechQSL(
            data_path="/tmp/dummy",
        )
        
        # Default is 2513 (MLPerf default)
        assert qsl._performance_sample_count == 2513


class TestMelSpectrogram:
    """Tests for mel spectrogram computation."""
    
    def test_log_mel_spectrogram_shape(self):
        """Test mel spectrogram output shape."""
        from mlperf_openvino.datasets.librispeech import (
            log_mel_spectrogram,
            N_MELS,
        )
        
        # Create dummy audio (1 second at 16kHz)
        audio = np.random.randn(16000).astype(np.float32)
        
        mel = log_mel_spectrogram(audio)
        
        assert mel.shape[0] == N_MELS
        assert mel.dtype == np.float32
    
    def test_log_mel_spectrogram_30_seconds(self):
        """Test mel spectrogram for 30 second audio."""
        from mlperf_openvino.datasets.librispeech import (
            log_mel_spectrogram,
            N_MELS,
            N_SAMPLES,
        )
        
        # 30 seconds of audio
        audio = np.random.randn(N_SAMPLES).astype(np.float32) * 0.1
        
        mel = log_mel_spectrogram(audio)
        
        assert mel.shape[0] == N_MELS
        # For 30 seconds with hop_length=160: ~3000 frames
        assert mel.shape[1] > 2900
        assert mel.shape[1] < 3100


class TestWhisperConfig:
    """Tests for Whisper configuration."""
    
    def test_default_whisper_config(self):
        """Test default Whisper configuration."""
        from mlperf_openvino.core.config import BenchmarkConfig, ModelType
        
        config = BenchmarkConfig.default_whisper()
        
        assert config.model.name == "Whisper-Large-v3"
        assert config.model.task == "speech_recognition"
        assert config.model.model_type == ModelType.WHISPER
        assert config.model.input_shape == [1, 80, 3000]
        assert config.dataset.name == "librispeech"
    
    def test_whisper_input_name(self):
        """Test Whisper input/output names."""
        from mlperf_openvino.core.config import BenchmarkConfig
        
        config = BenchmarkConfig.default_whisper()
        
        assert config.model.input_name == "input_features"
        assert config.model.output_name == "sequences"
    
    def test_whisper_offline_config(self):
        """Test Whisper Offline scenario configuration."""
        from mlperf_openvino.core.config import BenchmarkConfig, Scenario
        
        config = BenchmarkConfig.default_whisper()
        config.scenario = Scenario.OFFLINE
        
        scenario_config = config.get_scenario_config()
        assert scenario_config.min_query_count == 2513
        assert scenario_config.min_duration_ms == 60000
    
    def test_whisper_server_config(self):
        """Test Whisper Server scenario configuration."""
        from mlperf_openvino.core.config import BenchmarkConfig, Scenario
        
        config = BenchmarkConfig.default_whisper()
        config.scenario = Scenario.SERVER
        
        scenario_config = config.get_scenario_config()
        assert scenario_config.target_latency_ns == 1000000000  # 1 second


class TestWhisperModelDownloader:
    """Tests for Whisper model downloader."""
    
    def test_whisper_in_registry(self):
        """Test that Whisper is in model registry."""
        from mlperf_openvino.utils.model_downloader import MODEL_REGISTRY
        
        assert "whisper" in MODEL_REGISTRY
        assert "huggingface" in MODEL_REGISTRY["whisper"]
    
    def test_whisper_model_info(self):
        """Test Whisper model info in registry."""
        from mlperf_openvino.utils.model_downloader import MODEL_REGISTRY
        
        whisper_info = MODEL_REGISTRY["whisper"]
        assert whisper_info["description"] == "Whisper Large v3 for speech recognition"
        assert whisper_info["huggingface"]["model_id"] == "openai/whisper-large-v3"


class TestLibriSpeechDatasetMethods:
    """Tests for LibriSpeech dataset methods."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        from mlperf_openvino.datasets.librispeech import LibriSpeechDataset
        
        dataset = LibriSpeechDataset(
            data_path="/tmp/test",
            count=100,
            max_duration=30.0,
        )
        
        assert dataset.data_path == Path("/tmp/test")
        assert dataset.count == 100
        assert dataset.max_duration == 30.0
        assert not dataset.is_loaded
    
    def test_dataset_postprocess(self):
        """Test postprocess method."""
        from mlperf_openvino.datasets.librispeech import LibriSpeechDataset
        
        dataset = LibriSpeechDataset(data_path="/tmp/test")
        
        # Test with string results
        results = ["hello world", "test transcript"]
        processed = dataset.postprocess(results, [0, 1])
        
        assert processed == results
    
    def test_compute_accuracy_basic(self):
        """Test basic accuracy computation."""
        from mlperf_openvino.datasets.librispeech import LibriSpeechDataset
        
        dataset = LibriSpeechDataset(data_path="/tmp/test")
        
        predictions = ["HELLO WORLD", "EXACT MATCH"]
        ground_truth = ["HELLO WORLD", "EXACT MATCH"]
        
        accuracy = dataset.compute_accuracy(predictions, ground_truth)
        
        # Should have some accuracy metric
        assert "num_samples" in accuracy or "wer" in accuracy or "exact_match" in accuracy
