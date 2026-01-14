"""
C++ accelerated SUT for maximum throughput.

This module provides C++ implementations of the SUT that bypass
Python GIL limitations for maximum inference throughput.

Two SUT types:
- CppSUT (Server): async inference, batch=1, for Server scenario
- CppOfflineSUT (Offline): sync batch inference, for Offline scenario

Usage:
    from mlperf_openvino.cpp import CppSUT, CppOfflineSUT, CPP_AVAILABLE

    if CPP_AVAILABLE:
        # For Server mode
        server_sut = CppSUT(model_path, device="CPU")
        server_sut.load()

        # For Offline mode
        offline_sut = CppOfflineSUT(model_path, device="CPU", batch_size=32)
        offline_sut.load()
"""

CPP_AVAILABLE = False
CppSUT = None
CppOfflineSUT = None

try:
    from ._cpp_sut import CppSUT, CppOfflineSUT
    CPP_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ SUT extension not available: {e}. "
        "Using Python fallback (slower). "
        "To build C++ extension: ./build_cpp.sh"
    )

__all__ = ["CppSUT", "CppOfflineSUT", "CPP_AVAILABLE"]
