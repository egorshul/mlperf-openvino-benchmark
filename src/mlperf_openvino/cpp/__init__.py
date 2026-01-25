"""C++ accelerated SUT for maximum throughput."""

CPP_AVAILABLE = False
ResNetCppSUT = None
BertCppSUT = None
RetinaNetCppSUT = None
ResNetMultiDieCppSUT = None
BertMultiDieSUT = None

try:
    from ._cpp_sut import (
        ResNetCppSUT,
        BertCppSUT,
        RetinaNetCppSUT,
        ResNetMultiDieCppSUT,
        BertMultiDieSUT,
    )
    CPP_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ SUT extension not available: {e}. "
        "Using Python fallback. To build: ./build_cpp.sh"
    )

__all__ = [
    "ResNetCppSUT", "BertCppSUT", "RetinaNetCppSUT",
    "ResNetMultiDieCppSUT", "BertMultiDieSUT", "CPP_AVAILABLE",
]
