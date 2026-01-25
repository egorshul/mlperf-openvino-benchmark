"""Core components for MLPerf OpenVINO Benchmark."""

from .config import (
    BenchmarkConfig,
    DatasetConfig,
    ModelConfig,
    OpenVINOConfig,
    Scenario,
    ScenarioConfig,
    TestMode,
)
from .benchmark_runner import BenchmarkRunner

# Whisper SUT exports (optional - may fail if dependencies not installed)
try:
    from .whisper_sut import (
        WhisperOptimumSUT,
        WhisperSUT,
        WhisperNPUSUT,
        WhisperOptimumNPUSUT,
        WhisperEncoderOnlySUT,
    )
    _whisper_exports = [
        "WhisperOptimumSUT",
        "WhisperSUT",
        "WhisperNPUSUT",
        "WhisperOptimumNPUSUT",
        "WhisperEncoderOnlySUT",
    ]
except ImportError:
    _whisper_exports = []

__all__ = [
    "BenchmarkConfig",
    "DatasetConfig",
    "ModelConfig",
    "OpenVINOConfig",
    "Scenario",
    "ScenarioConfig",
    "TestMode",
    "BenchmarkRunner",
] + _whisper_exports
