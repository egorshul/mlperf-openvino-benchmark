"""
Base backend interface for MLPerf OpenVINO Benchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseBackend(ABC):
    """Abstract base class for inference backends."""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the backend.
        
        Args:
            model_path: Path to the model file
            **kwargs: Backend-specific options
        """
        self.model_path = model_path
        self.options = kwargs
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the given inputs.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        pass
    
    @abstractmethod
    def predict_batch(self, batch: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch: List of input dictionaries
            
        Returns:
            List of output dictionaries
        """
        pass
    
    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """Get list of input tensor names."""
        pass
    
    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """Get list of output tensor names."""
        pass
    
    @property
    @abstractmethod
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of input tensors."""
        pass
    
    @property
    @abstractmethod
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of output tensors."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warm up the model by running a few inference iterations.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        if not self._loaded:
            self.load()
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            dummy_inputs[name] = np.random.randn(*shape).astype(np.float32)
        
        # Run warmup iterations
        for _ in range(num_iterations):
            self.predict(dummy_inputs)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the backend and model.
        
        Returns:
            Dictionary with backend information
        """
        return {
            "backend": self.__class__.__name__,
            "model_path": self.model_path,
            "loaded": self._loaded,
            "input_names": self.input_names if self._loaded else [],
            "output_names": self.output_names if self._loaded else [],
            "input_shapes": self.input_shapes if self._loaded else {},
            "output_shapes": self.output_shapes if self._loaded else {},
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
