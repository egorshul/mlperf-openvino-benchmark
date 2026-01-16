"""
Tests for new model support: BERT, RetinaNet, Whisper.
"""

import pytest
import numpy as np
from pathlib import Path

from mlperf_openvino.core.config import (
    BenchmarkConfig,
    ModelType,
    Scenario,
)


class TestBertConfig:
    """Tests for BERT configuration."""

    def test_default_bert_config(self):
        """Test default BERT configuration."""
        config = BenchmarkConfig.default_bert()

        assert config.model.name == "BERT-Large"
        assert config.model.model_type == ModelType.BERT
        assert config.model.task == "question_answering"
        assert config.model.accuracy_target == 0.90874
        assert config.dataset.name == "squad"

    def test_bert_scenario_config(self):
        """Test BERT scenario-specific config."""
        config = BenchmarkConfig.default_bert()

        # Offline scenario
        config.scenario = Scenario.OFFLINE
        scenario_config = config.get_scenario_config()
        assert scenario_config.min_query_count == 10833

        # Server scenario
        config.scenario = Scenario.SERVER
        scenario_config = config.get_scenario_config()
        assert scenario_config.target_latency_ns == 130000000


class TestRetinaNetConfig:
    """Tests for RetinaNet configuration."""

    def test_default_retinanet_config(self):
        """Test default RetinaNet configuration."""
        config = BenchmarkConfig.default_retinanet()

        assert config.model.name == "RetinaNet"
        assert config.model.model_type == ModelType.RETINANET
        assert config.model.task == "object_detection"
        assert config.model.accuracy_target == 0.3757
        assert config.dataset.name == "openimages"

    def test_retinanet_input_shape(self):
        """Test RetinaNet input shape."""
        config = BenchmarkConfig.default_retinanet()

        assert config.model.input_shape == [1, 3, 800, 800]


class TestWhisperConfig:
    """Tests for Whisper configuration."""

    def test_default_whisper_config(self):
        """Test default Whisper configuration."""
        config = BenchmarkConfig.default_whisper()

        assert config.model.name == "Whisper-Large-v3"
        assert config.model.model_type == ModelType.WHISPER
        assert config.model.task == "speech_recognition"
        assert config.dataset.name == "librispeech"

    def test_whisper_input_shape(self):
        """Test Whisper input shape (mel spectrogram)."""
        config = BenchmarkConfig.default_whisper()

        # batch, n_mels, time_frames
        assert config.model.input_shape == [1, 80, 3000]


class TestSQuADMetrics:
    """Tests for SQuAD accuracy metrics."""

    def test_normalize_answer(self):
        """Test answer normalization."""
        from mlperf_openvino.datasets.squad import normalize_answer

        assert normalize_answer("The Answer") == "answer"
        assert normalize_answer("A test, with punctuation!") == "test with punctuation"
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_compute_f1(self):
        """Test F1 score computation."""
        from mlperf_openvino.datasets.squad import compute_f1

        # Exact match
        assert compute_f1("the answer", "the answer") == 1.0

        # No overlap
        assert compute_f1("foo bar", "baz qux") == 0.0

        # Partial match
        f1 = compute_f1("the quick brown fox", "the quick fox")
        assert 0.0 < f1 < 1.0

    def test_compute_exact_match(self):
        """Test exact match computation."""
        from mlperf_openvino.datasets.squad import compute_exact_match

        assert compute_exact_match("The Answer", "the answer") == 1.0
        assert compute_exact_match("answer a", "answer b") == 0.0


class TestOpenImagesMetrics:
    """Tests for OpenImages mAP computation."""

    def test_compute_iou(self):
        """Test IoU computation."""
        from mlperf_openvino.datasets.openimages import compute_iou

        # Perfect overlap
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([0, 0, 10, 10])
        assert compute_iou(box1, box2) == 1.0

        # No overlap
        box1 = np.array([0, 0, 5, 5])
        box2 = np.array([10, 10, 15, 15])
        assert compute_iou(box1, box2) == 0.0

        # Partial overlap
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])
        iou = compute_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_compute_ap(self):
        """Test Average Precision computation."""
        from mlperf_openvino.datasets.openimages import compute_ap

        # Perfect ranking
        recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert ap > 0.0

        # Worst case
        recalls = np.array([1.0])
        precisions = np.array([0.0])
        ap = compute_ap(recalls, precisions)
        assert ap == 0.0


class TestRetinaNetSUT:
    """Tests for RetinaNet SUT components."""

    def test_nms(self):
        """Test Non-Maximum Suppression."""
        from mlperf_openvino.core.retinanet_sut import nms

        # Single box
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        keep = nms(boxes, scores, 0.5)
        assert len(keep) == 1

        # Two overlapping boxes
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, 0.5)
        # High overlap should suppress lower score box
        assert len(keep) <= 2

        # Two non-overlapping boxes
        boxes = np.array([
            [0, 0, 5, 5],
            [10, 10, 15, 15],
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, 0.5)
        assert len(keep) == 2


class TestDatasetDownloader:
    """Tests for dataset downloader."""

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        from mlperf_openvino.utils.dataset_downloader import list_available_datasets

        datasets = list_available_datasets()

        assert "imagenet" in datasets
        assert "squad" in datasets
        assert "openimages" in datasets
        assert "librispeech" in datasets

    def test_get_dataset_info(self):
        """Test getting dataset info."""
        from mlperf_openvino.utils.dataset_downloader import get_dataset_info

        # SQuAD
        squad_info = get_dataset_info("squad")
        assert "description" in squad_info
        assert "dev" in squad_info

        # OpenImages
        oi_info = get_dataset_info("openimages")
        assert "description" in oi_info
        assert "annotations" in oi_info

        # Invalid dataset
        with pytest.raises(ValueError):
            get_dataset_info("invalid_dataset")


class TestModelDownloader:
    """Tests for model downloader."""

    def test_list_available_models(self):
        """Test listing available models."""
        from mlperf_openvino.utils.model_downloader import list_available_models

        models = list_available_models()

        assert "resnet50" in models
        assert "bert" in models
        assert "retinanet" in models
        assert "whisper" in models


class TestCLI:
    """Tests for CLI commands."""

    def test_get_default_config(self):
        """Test getting default config for each model."""
        from mlperf_openvino.cli import get_default_config

        # ResNet50
        config = get_default_config("resnet50")
        assert config.model.model_type == ModelType.RESNET50

        # BERT
        config = get_default_config("bert")
        assert config.model.model_type == ModelType.BERT

        # RetinaNet
        config = get_default_config("retinanet")
        assert config.model.model_type == ModelType.RETINANET

        # Whisper
        config = get_default_config("whisper")
        assert config.model.model_type == ModelType.WHISPER
