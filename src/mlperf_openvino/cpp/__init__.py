"""C++ accelerated SUT for maximum throughput."""

CPP_AVAILABLE = False
ResNetCppSUT = None
BertCppSUT = None
RetinaNetCppSUT = None
SSDResNet34CppSUT = None
ResNetMultiDieCppSUT = None
BertMultiDieSUT = None
RetinaNetMultiDieCppSUT = None
SSDResNet34MultiDieCppSUT = None
UNet3DMultiDieCppSUT = None

try:
    from ._cpp_sut import (
        ResNetCppSUT,
        BertCppSUT,
        RetinaNetCppSUT,
        SSDResNet34CppSUT,
        ResNetMultiDieCppSUT,
        BertMultiDieSUT,
        RetinaNetMultiDieCppSUT,
        SSDResNet34MultiDieCppSUT,
        UNet3DMultiDieCppSUT,
    )
    CPP_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ SUT extension not available: {e}. "
        "Using Python fallback. To build: ./build_cpp.sh"
    )

__all__ = [
    "ResNetCppSUT", "BertCppSUT", "RetinaNetCppSUT", "SSDResNet34CppSUT",
    "ResNetMultiDieCppSUT", "BertMultiDieSUT", "RetinaNetMultiDieCppSUT",
    "SSDResNet34MultiDieCppSUT", "UNet3DMultiDieCppSUT",
    "CPP_AVAILABLE",
]
