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

_feature_extractor = None


def get_whisper_feature_extractor():
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
            _feature_extractor = False
        except Exception as e:
            logger.warning(f"Failed to load WhisperFeatureExtractor: {e}")
            _feature_extractor = False
    return _feature_extractor if _feature_extractor else None


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio loading. "
            "Install with: pip install soundfile"
        )

    audio, file_sr = sf.read(file_path, dtype='float32')

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if file_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        except ImportError:
            ratio = sr / file_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

    return audio


def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
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
    try:
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=8000,
        )

        log_mel = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel.astype(np.float32)

    except ImportError:
        logger.warning("librosa not available, using simplified mel spectrogram")

        window = np.hanning(n_fft)
        num_frames = 1 + (len(audio) - n_fft) // hop_length

        stft = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            stft[:, i] = np.fft.rfft(frame)

        power = np.abs(stft) ** 2

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

        mel_spec = np.dot(filterbank, power)

        log_mel = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel.astype(np.float32)


class LibriSpeechDataset(BaseDataset):

    def __init__(
        self,
        data_path: str,
        transcript_path: Optional[str] = None,
        count: Optional[int] = None,
        max_duration: float = 30.0,
    ):
        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.transcript_path = transcript_path
        self.max_duration = max_duration

        self._samples: List[Dict[str, Any]] = []
        self._cache: Dict[int, np.ndarray] = {}
        self._is_loaded = False

    def load(self) -> None:
        if self._is_loaded:
            return

        logger.info(f"Loading LibriSpeech dataset from {self.data_path}")

        # Try JSON manifest first (MLPerf/MLCommons format)
        manifest_file = None
        manifest_names = [
            "dev-all-repack.json",
            "dev-all.json",
            "manifest.json",
            "dev-clean.json",
            "dev-other.json",
        ]

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
                    logger.warning("No transcript file found, scanning for audio files")

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

        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        logger.info(f"Loaded {len(self._samples)} audio samples")
        self._is_loaded = True

    def _load_from_manifest(self, manifest_file: Path) -> None:
        import json

        logger.info(f"Loading from manifest: {manifest_file}")

        with open(manifest_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        entries = []
        if content.startswith('['):
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
            audio_path = None

            # MLCommons format: files[0].fname
            if "files" in entry and isinstance(entry["files"], list) and entry["files"]:
                fname = entry["files"][0].get("fname", "")
                if fname:
                    audio_path = fname

            if not audio_path:
                audio_path = entry.get("audio_filepath", "")

            if not audio_path:
                audio_path = entry.get("audio_path", "")

            if not audio_path:
                logger.debug(f"No audio path found in entry: {entry}")
                continue

            if not Path(audio_path).is_absolute():
                manifest_dir = manifest_file.parent
                candidate = manifest_dir / audio_path
                if candidate.exists():
                    audio_path = str(candidate)
                else:
                    candidate = self.data_path / audio_path
                    if candidate.exists():
                        audio_path = str(candidate)
                    else:
                        audio_name = Path(audio_path).name
                        for subdir in ["audio", "wav", "flac", ""]:
                            if subdir:
                                candidate = self.data_path / subdir / audio_name
                            else:
                                candidate = self.data_path / audio_name
                            if candidate.exists():
                                audio_path = str(candidate)
                                break

            if not Path(audio_path).exists():
                logger.debug(f"Audio file not found: {audio_path}")
                continue

            transcript = entry.get("transcript", "")
            if not transcript:
                transcript = entry.get("text", "")
            if not transcript and "text_filepath" in entry:
                text_path = Path(entry["text_filepath"])
                if not text_path.is_absolute():
                    text_path = manifest_file.parent / text_path
                if text_path.exists():
                    transcript = text_path.read_text().strip()

            sample_id = entry.get("utterance_id", "")
            if not sample_id:
                sample_id = entry.get("id", Path(audio_path).stem)

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
        audio_extensions = {'.flac', '.wav', '.mp3', '.ogg', '.m4a'}

        for root, _, files in os.walk(self.data_path):
            for file in sorted(files):
                if Path(file).suffix.lower() in audio_extensions:
                    audio_path = Path(root) / file
                    audio_id = Path(file).stem

                    self._samples.append({
                        "id": audio_id,
                        "audio_path": str(audio_path),
                        "transcript": "",
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
        if index in self._cache:
            features = self._cache[index]
        else:
            sample = self._samples[index]
            audio = load_audio(sample["audio_path"])

            feature_extractor = get_whisper_feature_extractor()

            if feature_extractor is not None:
                inputs = feature_extractor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="np",
                )
                features = inputs["input_features"]
            else:
                features = pad_or_trim(audio, N_SAMPLES)
                features = log_mel_spectrogram(features)
                features = features[np.newaxis, ...]

            self._cache[index] = features

        if features.ndim == 2:
            features = features[np.newaxis, ...]

        return features, self._samples[index]["transcript"]

    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[str]]:
        mels = []
        transcripts = []

        for idx in indices:
            mel, transcript = self.get_sample(idx)
            mels.append(mel[0])
            transcripts.append(transcript)

        return np.stack(mels), transcripts

    def get_transcript(self, index: int) -> str:
        return self._samples[index]["transcript"]

    def get_audio_path(self, index: int) -> str:
        return self._samples[index]["audio_path"]

    def postprocess(
        self,
        results: Union[np.ndarray, List[str]],
        indices: List[int]
    ) -> List[str]:
        if isinstance(results, np.ndarray):
            return ["" for _ in indices]
        return results

    def compute_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Compute Word Accuracy = 1 - WER (MLPerf v5.1 official metric for Whisper)."""
        try:
            from jiwer import wer, cer
        except ImportError:
            logger.warning("jiwer not installed, computing simple accuracy")
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
            from transformers import WhisperTokenizer
            tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            normalizer = tokenizer.normalize
            logger.info("Using WhisperTokenizer.normalize for WER calculation (MLCommons standard)")
        except Exception as e1:
            try:
                from transformers.models.whisper.english_normalizer import BasicTextNormalizer
                normalizer = BasicTextNormalizer()
                logger.info("Using BasicTextNormalizer for WER calculation")
            except Exception as e2:
                logger.warning(
                    f"Text normalizer not available ({e1}), using basic normalization. "
                    "This may result in LOWER accuracy scores! "
                    "Install/upgrade transformers: pip install transformers>=4.30"
                )

        if normalizer is not None:
            predictions_norm = [normalizer(p) for p in predictions]
            ground_truth_norm = [normalizer(g) for g in ground_truth]
        else:
            # Fallback: basic normalization (lowercase, strip)
            predictions_norm = [p.lower().strip() for p in predictions]
            ground_truth_norm = [g.lower().strip() for g in ground_truth]

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
            "word_accuracy": word_accuracy,
            "wer": word_error_rate,
            "cer": char_error_rate,
            "num_samples": len(valid_pairs),
        }


class LibriSpeechQSL(QuerySampleLibrary):

    def __init__(
        self,
        data_path: str,
        transcript_path: Optional[str] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 1633,  # MLPerf official for Whisper
    ):
        super().__init__()

        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            transcript_path=transcript_path,
            count=count,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, np.ndarray] = {}

    def load(self) -> None:
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
        for idx in sample_indices:
            if idx not in self._loaded_samples:
                mel, _ = self.dataset.get_sample(idx)
                self._loaded_samples[idx] = mel

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, np.ndarray]:
        if sample_index in self._loaded_samples:
            mel = self._loaded_samples[sample_index]
        else:
            mel, _ = self.dataset.get_sample(sample_index)

        return {"input_features": mel}

    def get_label(self, sample_index: int) -> str:
        return self.dataset.get_transcript(sample_index)
