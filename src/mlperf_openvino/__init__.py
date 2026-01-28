"""MLPerf v5.1 OpenVINO Benchmark."""

__version__ = "1.0.0"
__mlperf_version__ = "5.1"

from .core.benchmark_runner import BenchmarkRunner
from .core.config import BenchmarkConfig

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "__version__",
    "__mlperf_version__",
]
