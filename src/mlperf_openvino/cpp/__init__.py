"""
C++ accelerated SUT for maximum throughput.

This module provides a C++ implementation of the SUT that bypasses
Python GIL limitations for maximum inference throughput.

Usage:
    from mlperf_openvino.cpp import CppSUT, CPP_AVAILABLE

    if CPP_AVAILABLE:
        sut = CppSUT(model_path, device="CPU")
        sut.load()
        # Use C++ accelerated inference
    else:
        # Fall back to Python implementation
        pass
"""

CPP_AVAILABLE = False
CppSUT = None

try:
    from ._cpp_sut import CppSUT
    CPP_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ SUT extension not available: {e}. "
        "Using Python fallback (slower). "
        "To build C++ extension: python -m mlperf_openvino.cpp.setup_cpp"
    )

__all__ = ["CppSUT", "CPP_AVAILABLE"]
