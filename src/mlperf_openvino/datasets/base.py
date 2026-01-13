"""
Base dataset interface for MLPerf OpenVINO Benchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    
    Datasets are responsible for:
    - Loading and preprocessing data
    - Providing samples for inference
    - Computing accuracy metrics
    """
    
    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset
            count: Number of samples to use (None = all)
            **kwargs: Dataset-specific options
        """
        self.data_path = data_path
        self.count = count
        self.options = kwargs
        
        self._loaded = False
        self._items: List[Any] = []
        self._labels: List[Any] = []
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset into memory."""
        pass
    
    @abstractmethod
    def get_sample(self, index: int) -> Tuple[np.ndarray, Any]:
        """
        Get a preprocessed sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (preprocessed_data, label)
        """
        pass
    
    @abstractmethod
    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[Any]]:
        """
        Get multiple preprocessed samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (batch_data, labels)
        """
        pass
    
    @abstractmethod
    def postprocess(
        self, 
        results: np.ndarray, 
        indices: List[int]
    ) -> List[Any]:
        """
        Postprocess inference results.
        
        Args:
            results: Raw inference output
            indices: Sample indices
            
        Returns:
            Postprocessed results
        """
        pass
    
    @abstractmethod
    def compute_accuracy(
        self, 
        predictions: List[Any], 
        labels: List[Any]
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics.
        
        Args:
            predictions: List of predictions
            labels: List of ground truth labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        pass
    
    @property
    def total_count(self) -> int:
        """Get total number of samples."""
        return len(self._items)
    
    @property
    def sample_count(self) -> int:
        """Get number of samples to use (respects count limit)."""
        if self.count is None:
            return self.total_count
        return min(self.count, self.total_count)
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._loaded
    
    def get_item_list(self) -> List[Any]:
        """Get list of all items (e.g., file paths)."""
        return self._items[:self.sample_count]
    
    def get_labels(self) -> List[Any]:
        """Get list of all labels."""
        return self._labels[:self.sample_count]
    
    def __len__(self) -> int:
        """Get dataset length."""
        return self.sample_count
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        """Get a sample by index."""
        return self.get_sample(index)


class QuerySampleLibrary(ABC):
    """
    MLPerf Query Sample Library interface.
    
    This class provides the interface expected by the MLPerf LoadGen.
    """
    
    @abstractmethod
    def load_query_samples(self, sample_list: List[int]) -> None:
        """
        Load samples into memory.
        
        Args:
            sample_list: List of sample indices to load
        """
        pass
    
    @abstractmethod
    def unload_query_samples(self, sample_list: List[int]) -> None:
        """
        Unload samples from memory.
        
        Args:
            sample_list: List of sample indices to unload
        """
        pass
    
    @abstractmethod
    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        """
        Get features for a sample.
        
        Args:
            sample_id: Sample index
            
        Returns:
            Dictionary of input features
        """
        pass
    
    @property
    @abstractmethod
    def total_sample_count(self) -> int:
        """Get total number of samples."""
        pass
    
    @property
    @abstractmethod
    def performance_sample_count(self) -> int:
        """Get number of samples for performance testing."""
        pass
