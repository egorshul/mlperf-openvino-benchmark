"""Base backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class BaseBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.options = kwargs
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def predict_batch(self, batch: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        pass

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        pass

    @property
    @abstractmethod
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def warmup(self, num_iterations: int = 10) -> None:
        if not self._loaded:
            self.load()

        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            dummy_inputs[name] = np.random.randn(*shape).astype(np.float32)

        for _ in range(num_iterations):
            self.predict(dummy_inputs)

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": self.__class__.__name__,
            "model_path": self.model_path,
            "loaded": self._loaded,
            "input_names": self.input_names if self._loaded else [],
            "output_names": self.output_names if self._loaded else [],
            "input_shapes": self.input_shapes if self._loaded else {},
            "output_shapes": self.output_shapes if self._loaded else {},
        }
