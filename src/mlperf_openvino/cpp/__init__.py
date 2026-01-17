"""
C++ accelerated SUT for maximum throughput.

This module provides C++ implementations of the SUT that bypass
Python GIL limitations for maximum inference throughput.

SUT types:
- ResNetCppSUT: async inference for ResNet50 (single float32 input)
- CppOfflineSUT: sync batch inference for Offline scenario
- BertCppSUT: async inference for BERT (3x int64 inputs, 2x float32 outputs)
- RetinaNetCppSUT: async inference for RetinaNet (1x float32 input, 3 outputs)
- WhisperCppSUT: encoder-decoder inference for Whisper (mel spectrogram input, token output)

Usage:
    from mlperf_openvino.cpp import ResNetCppSUT, BertCppSUT, RetinaNetCppSUT, WhisperCppSUT

    # For ResNet50
    resnet_sut = ResNetCppSUT(model_path, device="CPU")
    resnet_sut.load()

    # For BERT
    bert_sut = BertCppSUT(model_path, device="CPU")
    bert_sut.load()

    # For RetinaNet
    retinanet_sut = RetinaNetCppSUT(model_path, device="CPU")
    retinanet_sut.load()

    # For Whisper
    whisper_sut = WhisperCppSUT(encoder_path, decoder_path, device="CPU")
    whisper_sut.load()
"""

CPP_AVAILABLE = False
ResNetCppSUT = None
CppOfflineSUT = None
BertCppSUT = None
RetinaNetCppSUT = None
WhisperCppSUT = None

try:
    from ._cpp_sut import ResNetCppSUT, CppOfflineSUT, BertCppSUT, RetinaNetCppSUT, WhisperCppSUT
    CPP_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ SUT extension not available: {e}. "
        "Using Python fallback (slower). "
        "To build C++ extension: ./build_cpp.sh"
    )

__all__ = [
    "ResNetCppSUT", "CppOfflineSUT", "BertCppSUT", "RetinaNetCppSUT", "WhisperCppSUT",
    "CPP_AVAILABLE",
]
