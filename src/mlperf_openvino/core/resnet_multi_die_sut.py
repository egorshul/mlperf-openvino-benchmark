import logging
from typing import Any, Dict

import numpy as np

try:
    from ..cpp import ResNetMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and ResNetMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    ResNetMultiDieCppSUT = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .image_multi_die_sut_base import ImageMultiDieSUTBase

logger = logging.getLogger(__name__)


class ResNetMultiDieCppSUTWrapper(ImageMultiDieSUTBase):
    """ResNet multi-die SUT for NPU accelerators."""

    MODEL_NAME = "ResNet"
    DEFAULT_OFFLINE_BATCH_SIZE = 1
    DEFAULT_OFFLINE_NIREQ_MULTIPLIER = 4
    DEFAULT_SERVER_NIREQ_MULTIPLIER = 2
    DEFAULT_EXPLICIT_BATCH_SIZE = 4
    DEFAULT_BATCH_TIMEOUT_US = 500

    def _check_cpp_availability(self) -> None:
        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

    def _create_cpp_sut(
        self,
        model_path: str,
        device_prefix: str,
        batch_size: int,
        compile_props: Dict,
        use_nhwc: bool,
        nireq_multiplier: int
    ) -> Any:
        return ResNetMultiDieCppSUT(
            model_path,
            device_prefix,
            batch_size,
            compile_props,
            use_nhwc,
            nireq_multiplier
        )

    def get_predictions(self) -> Dict[int, Any]:
        cpp_preds = self._cpp_sut.get_predictions()
        return {idx: np.array(pred) for idx, pred in cpp_preds.items()}


def is_resnet_multi_die_cpp_available() -> bool:
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
