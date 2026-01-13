"""Inference backends for MLPerf OpenVINO Benchmark."""

from .base import BaseBackend
from .openvino_backend import OpenVINOBackend

__all__ = [
    "BaseBackend",
    "OpenVINOBackend",
]
