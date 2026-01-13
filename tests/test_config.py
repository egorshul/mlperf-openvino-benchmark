"""
Tests for configuration module.
"""

import pytest
import tempfile
from pathlib import Path

from mlperf_openvino.core.config import (
    BenchmarkConfig,
    ModelConfig,
    OpenVINOConfig,
    PreprocessingConfig,
    Scenario,
    TestMode,
    ModelType,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""
    
    def test_default_values(self):
        """Test default preprocessing config."""
        config = PreprocessingConfig()
        
        assert config.resize == (256, 256)
        assert config.center_crop == (224, 224)
        assert config.mean == (123.68, 116.78, 103.94)
        assert config.std == (1.0, 1.0, 1.0)
        assert config.channel_order == "RGB"
    
    def test_custom_values(self):
        """Test custom preprocessing config."""
        config = PreprocessingConfig(
            resize=(320, 320),
            center_crop=(300, 300),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            channel_order="BGR"
        )
        
        assert config.resize == (320, 320)
        assert config.center_crop == (300, 300)
        assert config.channel_order == "BGR"


class TestOpenVINOConfig:
    """Tests for OpenVINOConfig."""
    
    def test_default_values(self):
        """Test default OpenVINO config."""
        config = OpenVINOConfig()
        
        assert config.device == "CPU"
        assert config.num_streams == "AUTO"
        assert config.num_threads == 0
        assert config.performance_hint == "THROUGHPUT"
    
    def test_to_properties(self):
        """Test properties conversion."""
        config = OpenVINOConfig(
            num_streams="4",
            num_threads=8,
            performance_hint="THROUGHPUT"
        )
        
        props = config.to_properties()
        
        assert "NUM_STREAMS" in props or "INFERENCE_NUM_THREADS" in props
    
    def test_auto_detection(self):
        """Test auto detection settings."""
        config = OpenVINOConfig(
            num_streams="AUTO",
            num_threads=0
        )
        
        props = config.to_properties()
        
        # Auto values should either be omitted or set to AUTO
        assert props.get("NUM_STREAMS") == "AUTO" or "NUM_STREAMS" not in props


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_resnet50_config(self):
        """Test ResNet50 model config."""
        config = ModelConfig(
            name="ResNet50-v1.5",
            task="image_classification",
            model_type=ModelType.RESNET50,
            input_shape=[1, 3, 224, 224],
            accuracy_target=0.7646
        )
        
        assert config.name == "ResNet50-v1.5"
        assert config.task == "image_classification"
        assert config.input_shape == [1, 3, 224, 224]
        assert config.accuracy_target == 0.7646


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""
    
    def test_default_resnet50(self):
        """Test default ResNet50 configuration."""
        config = BenchmarkConfig.default_resnet50()
        
        assert config.model.name == "ResNet50-v1.5"
        assert config.model.model_type == ModelType.RESNET50
        assert config.model.accuracy_target == 0.7646
        assert config.scenario == Scenario.OFFLINE
        assert config.test_mode == TestMode.PERFORMANCE_ONLY
    
    def test_scenario_config(self):
        """Test getting scenario-specific config."""
        config = BenchmarkConfig.default_resnet50()
        
        # Test Offline scenario
        config.scenario = Scenario.OFFLINE
        scenario_config = config.get_scenario_config()
        assert scenario_config == config.model.offline
        
        # Test Server scenario
        config.scenario = Scenario.SERVER
        scenario_config = config.get_scenario_config()
        assert scenario_config == config.model.server
    
    def test_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
global:
  mlperf_version: "5.1"
  division: "open"

models:
  resnet50:
    name: "ResNet50-v1.5"
    task: "image_classification"
    input_shape: [1, 3, 224, 224]
    accuracy_target: 0.7646
    preprocessing:
      resize: [256, 256]
      center_crop: [224, 224]
    offline:
      min_duration_ms: 60000
      min_query_count: 24576

openvino:
  device: "CPU"
  num_streams: "AUTO"

datasets:
  imagenet2012:
    path: "./data/imagenet"

output:
  results_dir: "./results"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            config = BenchmarkConfig.from_yaml(yaml_path, "resnet50")
            
            assert config.mlperf_version == "5.1"
            assert config.model.name == "ResNet50-v1.5"
            assert config.model.accuracy_target == 0.7646
            assert config.openvino.device == "CPU"
        finally:
            Path(yaml_path).unlink()


class TestScenario:
    """Tests for Scenario enum."""
    
    def test_offline_scenario(self):
        """Test Offline scenario."""
        assert Scenario.OFFLINE.value == "Offline"
    
    def test_server_scenario(self):
        """Test Server scenario."""
        assert Scenario.SERVER.value == "Server"


class TestTestMode:
    """Tests for TestMode enum."""
    
    def test_accuracy_mode(self):
        """Test accuracy mode."""
        assert TestMode.ACCURACY_ONLY.value == "AccuracyOnly"
    
    def test_performance_mode(self):
        """Test performance mode."""
        assert TestMode.PERFORMANCE_ONLY.value == "PerformanceOnly"
