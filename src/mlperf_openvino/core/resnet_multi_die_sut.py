"""
Python wrapper for C++ ResNetMultiDieCppSUT.

Provides MLPerf LoadGen integration while using the high-performance
C++ backend for inference on multi-die accelerators.
"""

import array
import logging
import sys
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

try:
    from ..cpp import ResNetMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and ResNetMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    ResNetMultiDieCppSUT = None

from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class ResNetMultiDieCppSUTWrapper:
    """
    Wrapper for C++ multi-die SUT with MLPerf LoadGen integration.

    Uses high-performance C++ backend for maximum throughput on
    multi-die accelerators (NPU, VPU, etc.).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
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

        # For Server mode, always use batch_size=1 for optimal latency
        # For Offline mode, use configured batch_size for throughput
        if scenario == Scenario.SERVER:
            self.batch_size = 1
        else:
            self.batch_size = config.openvino.batch_size

        # Get input name from config
        self.input_name = config.model.input_name

        # Build compile properties from config
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = dict(config.openvino.device_properties)  # Make a copy

        # NOTE: Don't add AUTO_BATCH_TIMEOUT here - it adds latency to Server mode!
        # The accelerator should handle batching internally without timeout.

        # Check if using NHWC input layout
        use_nhwc = False
        if hasattr(config.model, 'preprocessing') and config.model.preprocessing:
            use_nhwc = getattr(config.model.preprocessing, 'output_layout', 'NCHW') == 'NHWC'

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = ResNetMultiDieCppSUT(
            config.model.model_path,
            device_prefix,
            self.batch_size,
            compile_props,
            use_nhwc
        )

        # Statistics
        self._start_time = 0.0
        self._query_count = 0

        # For Offline mode: accumulate responses
        self._offline_responses: List[tuple] = []
        self._offline_lock = threading.Lock()

        # Progress thread
        self._progress_stop = False
        self._progress_thread = None

        # Server mode: callback setup flag
        self._server_callback_set = False

        logger.debug(f"ResNetMultiDieCppSUTWrapper: device_prefix={device_prefix}, batch_size={self.batch_size}")

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load and compile the model.

        Args:
            is_accuracy_mode: If True, store predictions for accuracy validation.
                             For performance mode, set to False to reduce overhead.
        """
        self._cpp_sut.load()

        # Only store predictions in accuracy mode (reduces overhead in performance)
        self._cpp_sut.set_store_predictions(is_accuracy_mode)
        self._is_accuracy_mode = is_accuracy_mode

        # Get actual input name from C++ SUT
        self.input_name = self._cpp_sut.get_input_name()

        # Note: QSL data validation happens in _issue_query_server on first call
        # because QSL samples may not be loaded yet at this point
        self._qsl_validated = False

        logger.debug(
            f"C++ SUT loaded: {self._cpp_sut.get_num_dies()} dies, "
            f"{self._cpp_sut.get_total_requests()} total requests, "
            f"accuracy_mode={is_accuracy_mode}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._cpp_sut.is_loaded()

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _setup_response_callback(self, is_offline: bool) -> None:
        """Setup response callback for LoadGen."""
        if is_offline:
            # Offline: accumulate responses
            # For Offline mode, we just need to track query_ids - no output data needed
            # because accuracy is computed from stored predictions
            def batch_callback(query_ids: list):
                with self._offline_lock:
                    for qid in query_ids:
                        # Store query_id with empty response (data not needed for Offline)
                        self._offline_responses.append((qid, array.array('B', b'\x00')))
            self._cpp_sut.set_batch_response_callback(batch_callback)
        else:
            # Server: immediate response to LoadGen
            def batch_callback(query_ids: list):
                responses = [lg.QuerySampleResponse(qid, 0, 0) for qid in query_ids]
                lg.QuerySamplesComplete(responses)
            self._cpp_sut.set_batch_response_callback(batch_callback)

    def _start_progress_thread(self):
        """Start progress monitoring (silent)."""
        self._progress_stop = False
        self._start_time = time.time()
        self._progress_thread = None

    def _stop_progress_thread(self):
        """Stop progress thread."""
        self._progress_stop = True

    def _prevalidate_qsl_data(self) -> None:
        """Pre-validate and register QSL data pointers in C++ for Server mode.

        This registers data pointers directly in C++ to avoid Python overhead
        on the critical path.
        """
        self._cached_inputs = {}
        self._qsl_data_ready = False
        self._cpp_qsl_registered = False

        # Check if QSL has _loaded_samples
        if not hasattr(self.qsl, '_loaded_samples'):
            logger.warning("QSL does not have _loaded_samples, Server mode may be slower")
            return

        if not self.qsl._loaded_samples:
            return

        # Validate format of first sample
        first_idx = next(iter(self.qsl._loaded_samples.keys()))
        sample = self.qsl._loaded_samples[first_idx]

        # Check if already correct format
        if (isinstance(sample, np.ndarray) and
            sample.dtype == np.float32 and
            sample.flags['C_CONTIGUOUS'] and
            sample.ndim == 4):
            self._qsl_data_ready = True

            # Register all samples in C++ for fast dispatch
            try:
                for idx, data in self.qsl._loaded_samples.items():
                    self._cpp_sut.register_sample_data(idx, data)
                self._cpp_qsl_registered = True
                logger.debug(f"Registered {len(self.qsl._loaded_samples)} samples in C++ for fast dispatch")
            except Exception as e:
                logger.debug(f"Failed to register samples in C++: {e}")
                self._cpp_qsl_registered = False
        else:
            logger.debug(f"QSL data needs conversion: dtype={sample.dtype}, "
                       f"contiguous={sample.flags['C_CONTIGUOUS']}, ndim={sample.ndim}")

    def _get_input_data(self, sample_idx: int) -> np.ndarray:
        """Get input data for a sample."""
        if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
            return self.qsl._loaded_samples[sample_idx]
        else:
            features = self.qsl.get_features(sample_idx)
            return features.get("input", features.get(self.input_name))

    def _get_input_data_fast(self, sample_idx: int) -> np.ndarray:
        """Get input data with minimal processing (for Server mode).

        Assumes data is already validated and in correct format.
        """
        # Use cached if available
        if sample_idx in self._cached_inputs:
            return self._cached_inputs[sample_idx]

        # Fast path: data already in correct format
        if hasattr(self, '_qsl_data_ready') and self._qsl_data_ready:
            return self.qsl._loaded_samples[sample_idx]

        # Slow path: need conversion
        return self._get_input_data(sample_idx)

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode with batching."""
        self._start_time = time.time()
        self._offline_responses.clear()
        self._cpp_sut.reset_counters()

        self._setup_response_callback(is_offline=True)
        self._start_progress_thread()

        batch_size = self.batch_size
        num_samples = len(query_samples)

        logger.debug(f"C++ Offline: {num_samples} samples, batch_size={batch_size}")

        # Process in batches
        i = 0
        while i < num_samples:
            end_idx = min(i + batch_size, num_samples)
            actual_batch_size = end_idx - i

            # Collect batch
            query_ids = []
            sample_indices = []
            batch_data = []

            for j in range(i, end_idx):
                qs = query_samples[j]
                query_ids.append(qs.id)
                sample_indices.append(qs.index)

                input_data = self._get_input_data(qs.index)

                # Remove batch dim if present
                if input_data.ndim == 4 and input_data.shape[0] == 1:
                    input_data = input_data[0]

                batch_data.append(input_data)

            # Stack into batch
            if actual_batch_size == batch_size:
                batched_input = np.stack(batch_data, axis=0)
            else:
                # Pad incomplete batch
                padded = batch_data + [batch_data[-1]] * (batch_size - actual_batch_size)
                batched_input = np.stack(padded, axis=0)

            # Ensure float32 and contiguous
            if batched_input.dtype != np.float32:
                batched_input = batched_input.astype(np.float32)
            if not batched_input.flags['C_CONTIGUOUS']:
                batched_input = np.ascontiguousarray(batched_input)

            # Submit to C++ SUT
            self._cpp_sut.start_async_batch(
                batched_input,
                query_ids,
                sample_indices,
                actual_batch_size
            )

            i = end_idx

        # Wait for completion
        self._cpp_sut.wait_all()
        self._stop_progress_thread()

        # Send all responses to LoadGen
        responses = []
        for query_id, response_array in self._offline_responses:
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def _issue_query_server(self, query_samples: List) -> None:
        """Process queries in Server mode."""
        # Setup callback only once
        if not self._server_callback_set:
            self._setup_response_callback(is_offline=False)
            self._server_callback_set = True

        # Validate QSL data format on first call (samples are loaded by now)
        if not getattr(self, '_qsl_validated', False):
            self._prevalidate_qsl_data()
            self._qsl_validated = True

        # Check if we can use the fastest path (data pre-registered in C++)
        cpp_qsl_registered = getattr(self, '_cpp_qsl_registered', False)

        if cpp_qsl_registered:
            # FASTEST PATH: only pass query_ids and sample_indices
            query_ids = [qs.id for qs in query_samples]
            sample_indices = [qs.index for qs in query_samples]
            self._cpp_sut.issue_queries_server_fast(query_ids, sample_indices)
        else:
            # Pass data arrays to C++
            input_arrays = []
            query_ids = []
            sample_indices = []

            loaded_samples = getattr(self.qsl, '_loaded_samples', {})

            for qs in query_samples:
                if qs.index in loaded_samples:
                    input_data = loaded_samples[qs.index]
                else:
                    input_data = self._get_input_data(qs.index)

                # Ensure 4D with batch dim
                if input_data.ndim == 3:
                    input_data = input_data[np.newaxis, ...]

                # Ensure contiguous float32
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                if not input_data.flags['C_CONTIGUOUS']:
                    input_data = np.ascontiguousarray(input_data)

                input_arrays.append(input_data)
                query_ids.append(qs.id)
                sample_indices.append(qs.index)

            # Single call to C++ - GIL released during dispatch
            self._cpp_sut.issue_queries_server(input_arrays, query_ids, sample_indices)

        self._query_count += 1

    def issue_queries(self, query_samples: List) -> None:
        """Process incoming queries."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush pending queries."""
        self._cpp_sut.wait_all()
        self._stop_progress_thread()

    def get_sut(self):
        """Get LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

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
        return f"MultiDieCpp-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        """Return completed sample count for stats."""
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Any]:
        """Get stored predictions."""
        cpp_preds = self._cpp_sut.get_predictions()
        return {idx: np.array(pred) for idx, pred in cpp_preds.items()}

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._cpp_sut.set_store_predictions(store)

    def reset(self) -> None:
        """Reset state."""
        self._cpp_sut.reset_counters()
        self._offline_responses.clear()
        self._query_count = 0
        self._server_callback_set = False
        self._qsl_validated = False
        # Clear C++ sample cache
        try:
            self._cpp_sut.clear_sample_data()
        except Exception:
            pass
        self._cpp_qsl_registered = False
        self._qsl_data_ready = False


def is_resnet_multi_die_cpp_available() -> bool:
    """Check if C++ multi-die SUT is available."""
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
