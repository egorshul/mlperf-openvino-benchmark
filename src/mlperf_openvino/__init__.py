"""
MLPerf v5.1 OpenVINO Benchmark
==============================

A benchmark tool for measuring inference performance using OpenVINO backend,
compatible with MLPerf Inference v5.1 specifications.

Supported Devices:
- CPU: Intel processors with optimized threading
- X: Neural network accelerator with multi-die support

Supported Models:
- ResNet50 (Image Classification)
- BERT-Large (Question Answering)
- RetinaNet (Object Detection)
- Whisper Large v3 (Speech Recognition)
- SDXL (Text-to-Image Generation)

Supported Scenarios:
- Offline: Maximum throughput
- Server: Latency-constrained throughput
"""

__version__ = "0.2.0"
__mlperf_version__ = "5.1"

from .core.benchmark_runner import BenchmarkRunner
from .core.config import BenchmarkConfig

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "__version__",
    "__mlperf_version__",
]
