"""LibriSpeech dataset for Whisper ASR."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# Audio processing constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples for 30 seconds

# Global feature extractor (lazy loaded)
_feature_extractor = None


def get_whisper_feature_extractor():
    """Get or create WhisperFeatureExtractor for audio preprocessing."""
    global _feature_extractor
    if _feature_extractor is None:
        try:
            from transformers import WhisperFeatureExtractor
            _feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-large-v3"
            )
            logger.info("Using WhisperFeatureExtractor for audio preprocessing")
        except ImportError:
            logger.warning(
                "transformers not available, falling back to manual mel spectrogram. "
                "Install with: pip install transformers"
            )
            _feature_extractor = False  # Mark as unavailable
        except Exception as e:
            logger.warning(f"Failed to load WhisperFeatureExtractor: {e}")
            _feature_extractor = False
    return _feature_extractor if _feature_extractor else None


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Audio waveform as numpy array
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio loading. "
            "Install with: pip install soundfile"
        )
    
    audio, file_sr = sf.read(file_path, dtype='float32')
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if file_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        except ImportError:
            # Simple resampling without librosa
            ratio = sr / file_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return audio


def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """
    Pad or trim audio to exact length.
    
    Args:
        array: Audio waveform
        length: Target length in samples
        
    Returns:
        Padded or trimmed audio
    """
    if len(array) > length:
        array = array[:length]
    elif len(array) < length:
        array = np.pad(array, (0, length - len(array)))
    return array


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram from audio waveform.
    
    This matches the preprocessing used by Whisper.
    
    Args:
        audio: Audio waveform (16kHz)
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames)
    """
    try:
        import librosa
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=8000,
        )
        
        # Convert to log scale
        log_mel = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        
        # Normalize
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        
        return log_mel.astype(np.float32)
        
    except ImportError:
        # Fallback without librosa - simplified mel spectrogram
        logger.warning("librosa not available, using simplified mel spectrogram")
        
        # Compute STFT
        window = np.hanning(n_fft)
        num_frames = 1 + (len(audio) - n_fft) // hop_length
        
        stft = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            stft[:, i] = np.fft.rfft(frame)
        
        # Power spectrum
        power = np.abs(stft) ** 2
        
        # Simplified mel filterbank
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + SAMPLE_RATE / 2 / 700), n_mels + 2)
        mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
        
        fft_freqs = np.linspace(0, SAMPLE_RATE / 2, n_fft // 2 + 1)
        
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = mel_freqs[i]
            center = mel_freqs[i + 1]
            right = mel_freqs[i + 2]
            
            for j, freq in enumerate(fft_freqs):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        # Apply filterbank
        mel_spec = np.dot(filterbank, power)
        
        # Log scale
        log_mel = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        
        return log_mel.astype(np.float32)


class LibriSpeechDataset(BaseDataset):
    """
    LibriSpeech dataset for Whisper ASR benchmark.
    
    LibriSpeech is a corpus of read English speech derived from audiobooks.
    For MLPerf, typically the dev-clean or test-clean subset is used.
    
    Expected directory structure:
        data_path/
        ├── audio/
        │   ├── 1272-128104-0000.flac
        │   ├── 1272-128104-0001.flac
        │   └── ...
        └── transcripts.txt (or dev-clean.txt)
    
    Transcript format (one per line):
        1272-128104-0000 HE HOPED THERE WOULD BE STEW FOR DINNER
    """
    
    def __init__(
        self,
        data_path: str,
        transcript_path: Optional[str] = None,
        count: Optional[int] = None,
        max_duration: float = 30.0,
    ):
        """
        Initialize LibriSpeech dataset.
        
        Args:
            data_path: Path to dataset directory
            transcript_path: Path to transcript file (optional)
            count: Number of samples to use (None = all)
            max_duration: Maximum audio duration in seconds
        """
        super().__init__(data_path=data_path, count=count)
        
        self.data_path = Path(data_path)
        self.transcript_path = transcript_path
        self.max_duration = max_duration
        
        self._samples: List[Dict[str, Any]] = []
        self._cache: Dict[int, np.ndarray] = {}
        self._is_loaded = False
    
    def load(self) -> None:
        """Load dataset metadata."""
        if self._is_loaded:
            return

        logger.info(f"Loading LibriSpeech dataset from {self.data_path}")

        # Try JSON manifest first (MLPerf/MLCommons format)
        manifest_file = None
        manifest_names = [
            "dev-all-repack.json",  # MLCommons repackaged format
            "dev-all.json",         # MLCommons merged format
            "manifest.json",        # Generic manifest
            "dev-clean.json",       # Dev-clean split
            "dev-other.json",       # Dev-other split
        ]

        # Search in current directory and parent directory
        search_paths = [self.data_path]
        if self.data_path.parent != self.data_path:
            search_paths.append(self.data_path.parent)

        for search_path in search_paths:
            for name in manifest_names:
                candidate = search_path / name
                if candidate.exists():
                    manifest_file = candidate
                    logger.info(f"Found manifest: {manifest_file}")
                    break
            if manifest_file:
                break

        if manifest_file:
            self._load_from_manifest(manifest_file)
        else:
            # Fall back to transcript file
            if self.transcript_path:
                transcript_file = Path(self.transcript_path)
            else:
                # Try common names
                transcript_file = None
                transcript_names = [
                    "transcripts.txt",
                    "dev-clean.txt",
                    "dev-other.txt",
                    "test-clean.txt",
                ]
                for name in transcript_names:
                    candidate = self.data_path / name
                    if candidate.exists():
                        transcript_file = candidate
                        break

                if not transcript_file:
                    # Scan for audio files without transcripts
                    logger.warning("No transcript file found, scanning for audio files")

            # Load samples
            if transcript_file and transcript_file.exists():
                self._load_from_transcript(transcript_file)
            else:
                self._scan_audio_files()

        if not self._samples:
            logger.warning(f"No samples found in {self.data_path}")
            logger.info("For LibriSpeech dataset, expected structure:")
            logger.info("  data_path/")
            logger.info("  ├── dev-all-repack.json (MLCommons manifest)")
            logger.info("  └── *.wav or *.flac files")
            logger.info("")
            logger.info("Or use mlperf-ov download --model whisper to get the dataset")

        # Limit count if specified
        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        logger.info(f"Loaded {len(self._samples)} audio samples")
        self._is_loaded = True

    def _load_from_manifest(self, manifest_file: Path) -> None:
        """Load samples from JSON manifest file (MLPerf/MLCommons format).

        Supports multiple manifest formats:
        1. MLCommons format: {"files": [{"fname": "audio.wav"}], "transcript": "text"}
        2. NeMo format: {"audio_filepath": "audio.wav", "text": "text"}
        3. Simple format: {"audio_filepath": "audio.wav", "transcript": "text"}
        """
        import json

        logger.info(f"Loading from manifest: {manifest_file}")

        with open(manifest_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Try to parse as JSON array or JSONL (newline-delimited JSON)
        entries = []
        if content.startswith('['):
            # Standard JSON array
            entries = json.loads(content)
        else:
            # JSONL format (one JSON object per line)
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Found {len(entries)} entries in manifest")

        for entry in entries:
            # Extract audio path from various formats
            audio_path = None

            # MLCommons format: files[0].fname
            if "files" in entry and isinstance(entry["files"], list) and entry["files"]:
                fname = entry["files"][0].get("fname", "")
                if fname:
                    audio_path = fname

            # Standard format: audio_filepath
            if not audio_path:
                audio_path = entry.get("audio_filepath", "")

            # Alternative: audio_path directly
            if not audio_path:
                audio_path = entry.get("audio_path", "")

            if not audio_path:
                logger.debug(f"No audio path found in entry: {entry}")
                continue

            # Handle relative paths - resolve against manifest directory
            if not Path(audio_path).is_absolute():
                # First try relative to manifest file directory
                manifest_dir = manifest_file.parent
                candidate = manifest_dir / audio_path
                if candidate.exists():
                    audio_path = str(candidate)
                else:
                    # Try relative to data_path
                    candidate = self.data_path / audio_path
                    if candidate.exists():
                        audio_path = str(candidate)
                    else:
                        # Try in audio subdirectory
                        audio_name = Path(audio_path).name
                        for subdir in ["audio", "wav", "flac", ""]:
                            if subdir:
                                candidate = self.data_path / subdir / audio_name
                            else:
                                candidate = self.data_path / audio_name
                            if candidate.exists():
                                audio_path = str(candidate)
                                break

            # Skip if file doesn't exist
            if not Path(audio_path).exists():
                logger.debug(f"Audio file not found: {audio_path}")
                continue

            # Extract transcript from various fields
            transcript = entry.get("transcript", "")
            if not transcript:
                transcript = entry.get("text", "")
            if not transcript and "text_filepath" in entry:
                text_path = Path(entry["text_filepath"])
                if not text_path.is_absolute():
                    text_path = manifest_file.parent / text_path
                if text_path.exists():
                    transcript = text_path.read_text().strip()

            # Extract ID
            sample_id = entry.get("utterance_id", "")
            if not sample_id:
                sample_id = entry.get("id", Path(audio_path).stem)

            # Extract duration
            duration = entry.get("duration", 0.0)
            if not duration and "original_duration" in entry:
                duration = entry.get("original_duration", 0.0)

            self._samples.append({
                "id": sample_id,
                "audio_path": audio_path,
                "transcript": transcript,
                "duration": duration,
            })

        logger.info(f"Loaded {len(self._samples)} valid samples from manifest")
    
    def _load_from_transcript(self, transcript_file: Path) -> None:
        """Load samples from transcript file."""
        audio_dir = self.data_path / "audio"
        if not audio_dir.exists():
            audio_dir = self.data_path
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                
                audio_id = parts[0]
                transcript = parts[1]
                
                # Find audio file
                audio_file = None
                for ext in ['.flac', '.wav', '.mp3', '.ogg']:
                    candidate = audio_dir / f"{audio_id}{ext}"
                    if candidate.exists():
                        audio_file = candidate
                        break
                
                if audio_file:
                    self._samples.append({
                        "id": audio_id,
                        "audio_path": str(audio_file),
                        "transcript": transcript,
                    })
    
    def _scan_audio_files(self) -> None:
        """Scan directory for audio files."""
        audio_extensions = {'.flac', '.wav', '.mp3', '.ogg', '.m4a'}
        
        for root, _, files in os.walk(self.data_path):
            for file in sorted(files):
                if Path(file).suffix.lower() in audio_extensions:
                    audio_path = Path(root) / file
                    audio_id = Path(file).stem
                    
                    self._samples.append({
                        "id": audio_id,
                        "audio_path": str(audio_path),
                        "transcript": "",  # Unknown
                    })
    
    def __len__(self) -> int:
        return len(self._samples)
    
    @property
    def total_count(self) -> int:
        return len(self._samples)
    
    @property
    def sample_count(self) -> int:
        return len(self._samples)
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Get preprocessed audio sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (input_features, transcript)
            input_features shape: (1, n_mels, time_frames) - typically (1, 128, 3000) for Whisper
        """
        if index in self._cache:
            features = self._cache[index]
        else:
            sample = self._samples[index]
            audio = load_audio(sample["audio_path"])

            # Try to use WhisperFeatureExtractor for correct preprocessing
            feature_extractor = get_whisper_feature_extractor()

            if feature_extractor is not None:
                # Use HuggingFace feature extractor (produces correct shape for OpenVINO model)
                inputs = feature_extractor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="np",
                )
                features = inputs["input_features"]  # Shape: (1, 128, 3000) or (1, 80, 3000)
            else:
                # Fallback to manual mel spectrogram
                audio = pad_or_trim(audio, N_SAMPLES)
                mel = log_mel_spectrogram(audio)
                features = mel[np.newaxis, ...]  # (1, n_mels, time_frames)

            self._cache[index] = features

        # Ensure we have batch dimension
        if features.ndim == 2:
            features = features[np.newaxis, ...]

        return features, self._samples[index]["transcript"]
    
    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[str]]:
        """
        Get batch of preprocessed audio samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (mel_spectrograms, transcripts)
        """
        mels = []
        transcripts = []
        
        for idx in indices:
            mel, transcript = self.get_sample(idx)
            mels.append(mel[0])  # Remove batch dim for stacking
            transcripts.append(transcript)
        
        return np.stack(mels), transcripts
    
    def get_transcript(self, index: int) -> str:
        """Get transcript for sample."""
        return self._samples[index]["transcript"]
    
    def get_audio_path(self, index: int) -> str:
        """Get audio file path for sample."""
        return self._samples[index]["audio_path"]
    
    def postprocess(
        self,
        results: Union[np.ndarray, List[str]],
        indices: List[int]
    ) -> List[str]:
        """
        Postprocess model outputs.
        
        For Whisper, the output is typically decoded text.
        
        Args:
            results: Model outputs (decoded text or token IDs)
            indices: Sample indices
            
        Returns:
            List of transcribed texts
        """
        if isinstance(results, np.ndarray):
            # Assume token IDs - would need tokenizer for decoding
            # For now, return empty strings
            return ["" for _ in indices]
        return results
    
    def compute_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Compute Word Accuracy (MLPerf v5.1 official metric for Whisper).

        Word Accuracy = 1 - WER (Word Error Rate)

        Uses EnglishTextNormalizer from transformers for proper text normalization
        as required by MLCommons Whisper benchmark specification.

        Args:
            predictions: Predicted transcriptions
            ground_truth: Ground truth transcriptions

        Returns:
            Dictionary with Word Accuracy and other metrics
        """
        try:
            from jiwer import wer, cer
        except ImportError:
            logger.warning("jiwer not installed, computing simple accuracy")
            # Simple exact match accuracy
            correct = sum(
                1 for p, g in zip(predictions, ground_truth)
                if p.upper().strip() == g.upper().strip()
            )
            exact_match = correct / len(predictions) if predictions else 0.0
            return {
                "word_accuracy": exact_match,
                "exact_match": exact_match,
                "num_samples": len(predictions),
            }

        # Try to use Whisper's EnglishTextNormalizer (MLCommons standard)
        normalizer = None
        try:
            # Method 1: Use WhisperTokenizer's normalize method (recommended)
            from transformers import WhisperTokenizer
            tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            # The tokenizer has a normalize() method that uses EnglishTextNormalizer internally
            normalizer = tokenizer.normalize
            logger.info("Using WhisperTokenizer.normalize for WER calculation (MLCommons standard)")
        except Exception as e1:
            try:
                # Method 2: Try BasicTextNormalizer (doesn't need spelling mapping)
                from transformers.models.whisper.english_normalizer import BasicTextNormalizer
                normalizer = BasicTextNormalizer()
                logger.info("Using BasicTextNormalizer for WER calculation")
            except Exception as e2:
                logger.warning(
                    f"Text normalizer not available ({e1}), using basic normalization. "
                    "This may result in LOWER accuracy scores! "
                    "Install/upgrade transformers: pip install transformers>=4.30"
                )

        # Normalize texts
        if normalizer is not None:
            # Use Whisper's English normalizer (handles numbers, contractions, etc.)
            predictions_norm = [normalizer(p) for p in predictions]
            ground_truth_norm = [normalizer(g) for g in ground_truth]
        else:
            # Fallback: basic normalization (lowercase, strip whitespace)
            predictions_norm = [p.lower().strip() for p in predictions]
            ground_truth_norm = [g.lower().strip() for g in ground_truth]

        # Filter out empty references
        valid_pairs = [
            (p, g) for p, g in zip(predictions_norm, ground_truth_norm) if g
        ]

        if not valid_pairs:
            return {
                "word_accuracy": 0.0,
                "wer": 0.0,
                "cer": 0.0,
                "num_samples": 0,
            }

        preds, refs = zip(*valid_pairs)

        word_error_rate = wer(list(refs), list(preds))
        char_error_rate = cer(list(refs), list(preds))

        # MLPerf v5.1 uses Word Accuracy = 1 - WER
        word_accuracy = 1.0 - word_error_rate

        return {
            "word_accuracy": word_accuracy,  # Primary MLPerf metric
            "wer": word_error_rate,
            "cer": char_error_rate,
            "num_samples": len(valid_pairs),
        }


class LibriSpeechQSL(QuerySampleLibrary):
    """
    Query Sample Library for LibriSpeech dataset.
    
    Implements the MLPerf LoadGen QSL interface for Whisper benchmark.
    """
    
    def __init__(
        self,
        data_path: str,
        transcript_path: Optional[str] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 1633,  # MLPerf official for Whisper
    ):
        """
        Initialize LibriSpeech QSL.
        
        Args:
            data_path: Path to dataset directory
            transcript_path: Path to transcript file
            count: Number of samples to use
            performance_sample_count: Number of samples for performance run
        """
        super().__init__()
        
        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            transcript_path=transcript_path,
            count=count,
        )
        
        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, np.ndarray] = {}
    
    def load(self) -> None:
        """Load the dataset."""
        self.dataset.load()
    
    @property
    def total_sample_count(self) -> int:
        if not self.dataset._is_loaded:
            self.dataset.load()
        return self.dataset.total_count
    
    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)
    
    def load_query_samples(self, sample_indices: List[int]) -> None:
        """
        Load samples into memory.
        
        Args:
            sample_indices: Indices of samples to load
        """
        for idx in sample_indices:
            if idx not in self._loaded_samples:
                mel, _ = self.dataset.get_sample(idx)
                self._loaded_samples[idx] = mel
    
    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """
        Unload samples from memory.
        
        Args:
            sample_indices: Indices of samples to unload
        """
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)
    
    def get_features(self, sample_index: int) -> Dict[str, np.ndarray]:
        """
        Get input features for a sample.
        
        Args:
            sample_index: Sample index
            
        Returns:
            Dictionary with input features
        """
        if sample_index in self._loaded_samples:
            mel = self._loaded_samples[sample_index]
        else:
            mel, _ = self.dataset.get_sample(sample_index)
        
        return {"input_features": mel}
    
    def get_label(self, sample_index: int) -> str:
        """
        Get ground truth transcript for a sample.
        
        Args:
            sample_index: Sample index
            
        Returns:
            Ground truth transcript
        """
        return self.dataset.get_transcript(sample_index)
