"""
RetinaNet Multi-Die C++ SUT wrapper for MLPerf LoadGen.

This module wraps the C++ RetinaNetMultiDieCppSUT for use with MLPerf LoadGen,
providing maximum performance on multi-die accelerators.
"""

import array
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, Scenario
from ..datasets.openimages import OpenImagesQSL

try:
    from ..cpp import RetinaNetMultiDieCppSUT, CPP_AVAILABLE as CPP_SUT_AVAILABLE
except ImportError:
    CPP_SUT_AVAILABLE = False
    RetinaNetMultiDieCppSUT = None

logger = logging.getLogger(__name__)


class RetinaNetMultiDieCppSUTWrapper:
    """
    MLPerf LoadGen wrapper for C++ RetinaNet multi-die SUT.

    Uses high-performance C++ backend for maximum throughput on
    multi-die accelerators (NPU, VPU, etc.).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: OpenImagesQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen not installed")

        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Get input name from config
        self.input_name = config.model.input_name

        # Build compile properties from config
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = config.openvino.device_properties

        # Check if using NHWC input layout
        use_nhwc = False
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NCHW') == 'NHWC'

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = RetinaNetMultiDieCppSUT(
            config.model.model_path,
            device_prefix,
            compile_props,
            use_nhwc
        )

        # Statistics
        self._start_time = 0.0
        self._query_count = 0

        # For Offline mode: accumulate responses
        self._offline_responses: List[tuple] = []
        self._offline_lock = threading.Lock()

        # Server mode: callback setup flag
        self._server_callback_set = False

        logger.debug(f"RetinaNetMultiDieCppSUTWrapper: device_prefix={device_prefix}")

    def load(self) -> None:
        """Load and compile the model."""
        self._cpp_sut.load()

        # Enable prediction storage for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Get actual input name from C++ SUT
        self.input_name = self._cpp_sut.get_input_name()

        logger.debug(
            f"RetinaNet C++ SUT loaded: {self._cpp_sut.get_num_dies()} dies, "
            f"{self._cpp_sut.get_total_requests()} total requests"
        )

    @property
    def is_loaded(self) -> bool:
        return self._cpp_sut.is_loaded()

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _setup_response_callback(self, is_offline: bool = False):
        """Set up LoadGen response callback."""
        if is_offline:
            # For Offline: accumulate responses
            def callback(query_id, boxes, scores, labels):
                with self._offline_lock:
                    # Create response with minimal data (LoadGen doesn't use the data in Offline)
                    response = lg.QuerySampleResponse(query_id, 0, 0)
                    self._offline_responses.append(response)

            self._cpp_sut.set_response_callback(callback)
        else:
            # For Server: respond immediately
            def callback(query_id, boxes, scores, labels):
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

            self._cpp_sut.set_response_callback(callback)

    def _get_input_data(self, sample_idx: int) -> np.ndarray:
        """Get input data for a sample."""
        if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
            return self.qsl._loaded_samples[sample_idx]
        else:
            features = self.qsl.get_features(sample_idx)
            return features.get("input", features.get(self.input_name))

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode."""
        self._start_time = time.time()
        self._offline_responses.clear()
        self._cpp_sut.reset_counters()

        self._setup_response_callback(is_offline=True)

        num_samples = len(query_samples)
        logger.debug(f"RetinaNet C++ Offline: {num_samples} samples")

        # Process each sample
        for qs in query_samples:
            input_data = self._get_input_data(qs.index)

            # Remove batch dimension if present
            if input_data.ndim == 4 and input_data.shape[0] == 1:
                input_data = input_data.squeeze(0)

            # Ensure contiguous
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            self._cpp_sut.start_async(input_data, qs.id, qs.index)

        # Wait for all to complete
        self._cpp_sut.wait_all()

        # Submit all responses at once
        with self._offline_lock:
            if self._offline_responses:
                lg.QuerySamplesComplete(self._offline_responses)
                self._offline_responses.clear()

        self._query_count = num_samples

    def _issue_query_server(self, query_samples: List) -> None:
        """Process queries in Server mode."""
        # Setup callback only once
        if not self._server_callback_set:
            self._setup_response_callback(is_offline=False)
            self._server_callback_set = True

        for qs in query_samples:
            input_data = self._get_input_data(qs.index)

            if input_data.ndim == 4 and input_data.shape[0] == 1:
                input_data = input_data.squeeze(0)

            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            self._cpp_sut.start_async(input_data, qs.id, qs.index)

        self._query_count += len(query_samples)

    def issue_queries(self, query_samples: List) -> None:
        """Issue queries to the SUT."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        else:
            self._issue_query_server(query_samples)

    def flush_queries(self) -> None:
        """Flush pending queries."""
        self._cpp_sut.wait_all()

    def get_sut(self):
        """Get LoadGen SUT object."""
        return lg.ConstructSUT(
            self.issue_queries,
            self.flush_queries,
        )

    def get_qsl(self):
        """Get LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"RetinaNetMultiDieCpp-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        """Return completed sample count for stats."""
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Any]:
        """Get stored predictions."""
        return self._cpp_sut.get_predictions()

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._cpp_sut.set_store_predictions(store)

    def reset(self) -> None:
        """Reset state."""
        self._cpp_sut.reset_counters()
        self._offline_responses.clear()
        self._query_count = 0
        self._server_callback_set = False


def is_retinanet_multi_die_cpp_available() -> bool:
    """Check if C++ multi-die SUT is available."""
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
