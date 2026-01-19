"""
ResNet50 System Under Test (SUT) implementation for MLPerf.

This module provides the SUT implementation for ResNet50 image classification
using OpenVINO backend.
"""

import logging
from typing import Any, List

import numpy as np

from .sut import OpenVINOSUT
from .config import BenchmarkConfig, Scenario
from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.base import QuerySampleLibrary

logger = logging.getLogger(__name__)


class ResNetSUT(OpenVINOSUT):
    """
    ResNet50 System Under Test implementation.

    Inherits from OpenVINOSUT and provides ResNet50-specific functionality
    for image classification on ImageNet.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: OpenVINOBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize ResNet50 SUT.

        Args:
            config: Benchmark configuration
            backend: OpenVINO backend instance
            qsl: Query Sample Library (ImageNetQSL)
            scenario: Test scenario (Offline or Server)
        """
        super().__init__(config, backend, qsl, scenario)
        logger.info("ResNet50 SUT initialized")

    def _process_result(self, output: np.ndarray) -> int:
        """
        Process ResNet50 output to get predicted class.

        Args:
            output: Model output tensor

        Returns:
            Predicted class index
        """
        return int(np.argmax(output))

    def get_prediction(self, sample_idx: int) -> int:
        """
        Get prediction for a sample.

        Args:
            sample_idx: Sample index

        Returns:
            Predicted class index
        """
        features = self.qsl.get_features(sample_idx)
        input_data = {self.input_name: features}
        output = self.backend.predict(input_data)
        return self._process_result(output[self.output_name])
