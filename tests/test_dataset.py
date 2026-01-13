"""
Tests for ImageNet dataset.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Check if PIL is available
pil_available = True
try:
    from PIL import Image
except ImportError:
    pil_available = False


class TestPreprocessingConfig:
    """Tests for preprocessing configuration."""
    
    def test_default_preprocessing(self):
        """Test default preprocessing values."""
        from mlperf_openvino.core.config import PreprocessingConfig
        
        config = PreprocessingConfig()
        
        assert config.resize == (256, 256)
        assert config.center_crop == (224, 224)
        assert len(config.mean) == 3
        assert len(config.std) == 3


@pytest.mark.skipif(not pil_available, reason="Pillow not installed")
class TestImageNetDataset:
    """Tests for ImageNet dataset."""
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for testing."""
        from PIL import Image
        
        # Create test images
        img_dir = tmp_path / "val"
        img_dir.mkdir()
        
        for i in range(5):
            img = Image.new('RGB', (256, 256), color=(i * 50, i * 50, i * 50))
            img.save(img_dir / f"ILSVRC2012_val_{i:08d}.JPEG")
        
        # Create val_map.txt
        val_map = tmp_path / "val_map.txt"
        with open(val_map, 'w') as f:
            for i in range(5):
                f.write(f"ILSVRC2012_val_{i:08d}.JPEG {i}\n")
        
        return tmp_path
    
    def test_dataset_load(self, sample_images):
        """Test dataset loading."""
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        
        dataset = ImageNetDataset(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt")
        )
        
        dataset.load()
        
        assert dataset.is_loaded
        assert len(dataset) == 5
    
    def test_dataset_preprocessing(self, sample_images):
        """Test image preprocessing."""
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        from mlperf_openvino.core.config import PreprocessingConfig
        
        config = PreprocessingConfig(
            resize=(256, 256),
            center_crop=(224, 224),
            mean=(123.68, 116.78, 103.94),
            std=(1.0, 1.0, 1.0)
        )
        
        dataset = ImageNetDataset(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt"),
            preprocessing=config
        )
        
        dataset.load()
        
        # Get a sample
        sample, label = dataset.get_sample(0)
        
        # Check shape (NCHW format)
        assert sample.shape == (1, 3, 224, 224)
        assert sample.dtype == np.float32
    
    def test_batch_loading(self, sample_images):
        """Test batch loading."""
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        
        dataset = ImageNetDataset(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt")
        )
        
        dataset.load()
        
        # Get batch
        batch, labels = dataset.get_samples([0, 1, 2])
        
        assert batch.shape == (3, 3, 224, 224)
        assert len(labels) == 3
    
    def test_accuracy_computation(self, sample_images):
        """Test accuracy computation."""
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        
        dataset = ImageNetDataset(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt")
        )
        
        dataset.load()
        
        # Mock predictions
        predictions = [0, 1, 2, 3, 0]  # 4 correct
        labels = [0, 1, 2, 3, 4]
        
        accuracy = dataset.compute_accuracy(predictions, labels)
        
        assert accuracy["correct"] == 4
        assert accuracy["total"] == 5
        assert accuracy["top1_accuracy"] == 0.8


@pytest.mark.skipif(not pil_available, reason="Pillow not installed")
class TestImageNetQSL:
    """Tests for ImageNet Query Sample Library."""
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for testing."""
        from PIL import Image
        
        img_dir = tmp_path / "val"
        img_dir.mkdir()
        
        for i in range(10):
            img = Image.new('RGB', (256, 256), color=(i * 25, i * 25, i * 25))
            img.save(img_dir / f"ILSVRC2012_val_{i:08d}.JPEG")
        
        val_map = tmp_path / "val_map.txt"
        with open(val_map, 'w') as f:
            for i in range(10):
                f.write(f"ILSVRC2012_val_{i:08d}.JPEG {i}\n")
        
        return tmp_path
    
    def test_qsl_creation(self, sample_images):
        """Test QSL creation."""
        from mlperf_openvino.datasets.imagenet import ImageNetQSL
        
        qsl = ImageNetQSL(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt"),
            performance_sample_count=5
        )
        
        qsl.load()
        
        assert qsl.total_sample_count == 10
        assert qsl.performance_sample_count == 5
    
    def test_qsl_load_samples(self, sample_images):
        """Test sample loading."""
        from mlperf_openvino.datasets.imagenet import ImageNetQSL
        
        qsl = ImageNetQSL(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt")
        )
        
        qsl.load()
        qsl.load_query_samples([0, 1, 2])
        
        # Get features
        features = qsl.get_features(0)
        
        assert "input" in features
        assert features["input"].shape == (1, 3, 224, 224)
    
    def test_qsl_unload_samples(self, sample_images):
        """Test sample unloading."""
        from mlperf_openvino.datasets.imagenet import ImageNetQSL
        
        qsl = ImageNetQSL(
            data_path=str(sample_images),
            val_map_path=str(sample_images / "val_map.txt")
        )
        
        qsl.load()
        qsl.load_query_samples([0, 1, 2])
        
        # Verify loaded
        assert 0 in qsl._loaded_samples
        
        # Unload
        qsl.unload_query_samples([0, 1, 2])
        
        # Verify unloaded
        assert 0 not in qsl._loaded_samples


class TestPostprocessing:
    """Tests for result postprocessing."""
    
    @pytest.mark.skipif(not pil_available, reason="Pillow not installed")
    def test_postprocess_single(self, tmp_path):
        """Test postprocessing single result."""
        from PIL import Image
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        
        # Create test data
        img_dir = tmp_path / "val"
        img_dir.mkdir()
        img = Image.new('RGB', (256, 256))
        img.save(img_dir / "test.JPEG")
        
        val_map = tmp_path / "val_map.txt"
        with open(val_map, 'w') as f:
            f.write("test.JPEG 42\n")
        
        dataset = ImageNetDataset(
            data_path=str(tmp_path),
            val_map_path=str(val_map)
        )
        dataset.load()
        
        # Mock inference result
        result = np.zeros(1000, dtype=np.float32)
        result[42] = 10.0  # High score for class 42
        
        predicted = dataset.postprocess(result, [0])
        
        assert predicted[0] == 42
    
    @pytest.mark.skipif(not pil_available, reason="Pillow not installed")
    def test_postprocess_batch(self, tmp_path):
        """Test postprocessing batch results."""
        from PIL import Image
        from mlperf_openvino.datasets.imagenet import ImageNetDataset
        
        # Create test data
        img_dir = tmp_path / "val"
        img_dir.mkdir()
        
        for i in range(3):
            img = Image.new('RGB', (256, 256))
            img.save(img_dir / f"test_{i}.JPEG")
        
        val_map = tmp_path / "val_map.txt"
        with open(val_map, 'w') as f:
            for i in range(3):
                f.write(f"test_{i}.JPEG {i}\n")
        
        dataset = ImageNetDataset(
            data_path=str(tmp_path),
            val_map_path=str(val_map)
        )
        dataset.load()
        
        # Mock batch inference result
        results = np.zeros((3, 1000), dtype=np.float32)
        results[0, 0] = 10.0
        results[1, 1] = 10.0
        results[2, 2] = 10.0
        
        predicted = dataset.postprocess(results, [0, 1, 2])
        
        assert predicted == [0, 1, 2]
