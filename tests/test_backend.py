"""
Tests for OpenVINO backend.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Skip tests if OpenVINO is not available
openvino_available = True
try:
    import openvino as ov
except ImportError:
    openvino_available = False

pytestmark = pytest.mark.skipif(
    not openvino_available,
    reason="OpenVINO not installed"
)


class TestOpenVINOBackend:
    """Tests for OpenVINO backend."""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a dummy model for testing."""
        # This would normally require a real model file
        # For now, we'll skip actual model tests
        return str(tmp_path / "model.onnx")
    
    def test_backend_import(self):
        """Test that backend can be imported."""
        from mlperf_openvino.backends.openvino_backend import OpenVINOBackend, OPENVINO_AVAILABLE
        
        assert OPENVINO_AVAILABLE is True
    
    def test_backend_initialization(self):
        """Test backend initialization without loading."""
        from mlperf_openvino.backends.openvino_backend import OpenVINOBackend
        from mlperf_openvino.core.config import OpenVINOConfig
        
        config = OpenVINOConfig(
            device="CPU",
            num_threads=4,
            performance_hint="THROUGHPUT"
        )
        
        # Create backend (won't load without valid model path)
        backend = OpenVINOBackend(
            model_path="/nonexistent/model.onnx",
            config=config
        )
        
        assert backend.config.device == "CPU"
        assert backend.config.num_threads == 4
        assert not backend.is_loaded
    
    def test_config_to_properties(self):
        """Test OpenVINO config to properties conversion."""
        from mlperf_openvino.core.config import OpenVINOConfig
        
        config = OpenVINOConfig(
            device="CPU",
            num_streams="4",
            num_threads=8,
            performance_hint="THROUGHPUT",
            bind_thread=True
        )
        
        props = config.to_properties()
        
        # Check that properties are set correctly
        assert isinstance(props, dict)


class TestOpenVINOConfigProperties:
    """Tests for OpenVINO configuration properties."""
    
    def test_throughput_hint(self):
        """Test THROUGHPUT performance hint."""
        from mlperf_openvino.core.config import OpenVINOConfig
        
        config = OpenVINOConfig(performance_hint="THROUGHPUT")
        assert config.performance_hint == "THROUGHPUT"
    
    def test_latency_hint(self):
        """Test LATENCY performance hint."""
        from mlperf_openvino.core.config import OpenVINOConfig
        
        config = OpenVINOConfig(performance_hint="LATENCY")
        assert config.performance_hint == "LATENCY"
    
    def test_cpu_binding(self):
        """Test CPU thread binding option."""
        from mlperf_openvino.core.config import OpenVINOConfig
        
        config = OpenVINOConfig(bind_thread=True)
        assert config.bind_thread is True
        
        config = OpenVINOConfig(bind_thread=False)
        assert config.bind_thread is False


class TestOpenVINOBackendIntegration:
    """Integration tests that require a real model."""
    
    @pytest.fixture
    def simple_onnx_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            pytest.skip("ONNX not installed")
        
        # Create a simple identity model
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
        
        identity_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output'],
        )
        
        graph = helper.make_graph(
            [identity_node],
            'simple_model',
            [X],
            [Y]
        )
        
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        
        model_path = tmp_path / "simple_model.onnx"
        onnx.save(model, str(model_path))
        
        return str(model_path)
    
    def test_load_simple_model(self, simple_onnx_model):
        """Test loading a simple ONNX model."""
        from mlperf_openvino.backends.openvino_backend import OpenVINOBackend
        
        backend = OpenVINOBackend(model_path=simple_onnx_model)
        backend.load()
        
        assert backend.is_loaded
        assert len(backend.input_names) == 1
        assert len(backend.output_names) == 1
    
    def test_inference_simple_model(self, simple_onnx_model):
        """Test inference with a simple ONNX model."""
        from mlperf_openvino.backends.openvino_backend import OpenVINOBackend
        
        backend = OpenVINOBackend(model_path=simple_onnx_model)
        backend.load()
        
        # Create input
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        inputs = {backend.input_names[0]: input_data}
        
        # Run inference
        outputs = backend.predict(inputs)
        
        assert len(outputs) == 1
        output_data = list(outputs.values())[0]
        
        # Identity model should return same shape
        assert output_data.shape == input_data.shape
    
    def test_benchmark_simple_model(self, simple_onnx_model):
        """Test latency benchmark with simple model."""
        from mlperf_openvino.backends.openvino_backend import OpenVINOBackend
        
        backend = OpenVINOBackend(model_path=simple_onnx_model)
        backend.load()
        
        results = backend.benchmark(num_iterations=10, warmup_iterations=2)
        
        assert "mean_latency_ms" in results
        assert "throughput_fps" in results
        assert results["mean_latency_ms"] > 0
        assert results["throughput_fps"] > 0
