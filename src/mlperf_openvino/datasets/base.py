from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseDataset(ABC):

    def __init__(self, data_path: str, count: Optional[int] = None, **kwargs):
        self.data_path = data_path
        self.count = count
        self.options = kwargs

        self._loaded = False
        self._items: List[Any] = []
        self._labels: List[Any] = []

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def get_sample(self, index: int) -> Tuple[np.ndarray, Any]:
        pass

    @abstractmethod
    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[Any]]:
        pass

    @abstractmethod
    def postprocess(self, results: np.ndarray, indices: List[int]) -> List[Any]:
        pass

    @abstractmethod
    def compute_accuracy(self, predictions: List[Any], labels: List[Any]) -> Dict[str, float]:
        pass

    @property
    def total_count(self) -> int:
        return len(self._items)

    @property
    def sample_count(self) -> int:
        if self.count is None:
            return self.total_count
        return min(self.count, self.total_count)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_item_list(self) -> List[Any]:
        return self._items[:self.sample_count]

    def get_labels(self) -> List[Any]:
        return self._labels[:self.sample_count]

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        return self.get_sample(index)


class QuerySampleLibrary(ABC):

    @abstractmethod
    def load_query_samples(self, sample_list: List[int]) -> None:
        pass

    @abstractmethod
    def unload_query_samples(self, sample_list: List[int]) -> None:
        pass

    @abstractmethod
    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        pass

    @property
    @abstractmethod
    def total_sample_count(self) -> int:
        pass

    @property
    @abstractmethod
    def performance_sample_count(self) -> int:
        pass
