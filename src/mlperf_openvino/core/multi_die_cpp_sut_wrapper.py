"""
Python wrapper for C++ MultiDieCppSUT.

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
    from .._cpp_sut import MultiDieCppSUT
    CPP_SUT_AVAILABLE = True
except ImportError:
    CPP_SUT_AVAILABLE = False
    MultiDieCppSUT = None

from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class MultiDieCppSUTWrapper:
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
        self.batch_size = config.openvino.batch_size

        # Get input name from config
        self.input_name = config.model.input_name

        # Build compile properties from config
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = config.openvino.device_properties

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = MultiDieCppSUT(
            config.model.model_path,
            device_prefix,
            self.batch_size,
            compile_props
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

        logger.info(f"MultiDieCppSUTWrapper: device_prefix={device_prefix}, batch_size={self.batch_size}")

    def load(self) -> None:
        """Load and compile the model."""
        self._cpp_sut.load()

        # Get actual input name from C++ SUT
        self.input_name = self._cpp_sut.get_input_name()

        logger.info(
            f"C++ SUT loaded: {self._cpp_sut.get_num_dies()} dies, "
            f"{self._cpp_sut.get_total_requests()} total requests"
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
            def callback(query_id: int, output: np.ndarray):
                if output is not None:
                    response_array = array.array('B', output.tobytes())
                    with self._offline_lock:
                        self._offline_responses.append((query_id, response_array))
        else:
            # Server: immediate response
            def callback(query_id: int, output: np.ndarray):
                if output is not None:
                    response = lg.QuerySampleResponse(
                        query_id,
                        output.ctypes.data,
                        output.nbytes
                    )
                    lg.QuerySamplesComplete([response])
                else:
                    response = lg.QuerySampleResponse(query_id, 0, 0)
                    lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(callback)

    def _start_progress_thread(self):
        """Start progress monitoring."""
        self._progress_stop = False
        self._start_time = time.time()

        def progress_fn():
            while not self._progress_stop:
                time.sleep(0.3)
                elapsed = time.time() - self._start_time
                completed = self._cpp_sut.get_completed_count()
                issued = self._cpp_sut.get_issued_count()
                if elapsed > 0 and completed > 0:
                    throughput = completed / elapsed
                    pending = issued - completed
                    sys.stderr.write(
                        f"\r[C++ {self.num_dies} dies] "
                        f"done={completed}, pending={pending}, "
                        f"{throughput:.1f} samples/sec   "
                    )
                    sys.stderr.flush()

        self._progress_thread = threading.Thread(target=progress_fn, daemon=True)
        self._progress_thread.start()

    def _stop_progress_thread(self):
        """Stop progress thread."""
        self._progress_stop = True
        if self._progress_thread:
            self._progress_thread.join(timeout=1.0)

        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        if elapsed > 0 and completed > 0:
            throughput = completed / elapsed
            sys.stderr.write(
                f"\nC++ SUT completed: {completed} samples in {elapsed:.2f}s "
                f"({throughput:.1f} samples/sec)\n"
            )
            sys.stderr.flush()

    def _get_input_data(self, sample_idx: int) -> np.ndarray:
        """Get input data for a sample."""
        if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
            return self.qsl._loaded_samples[sample_idx]
        else:
            features = self.qsl.get_features(sample_idx)
            return features.get("input", features.get(self.input_name))

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode with batching."""
        self._start_time = time.time()
        self._offline_responses.clear()
        self._cpp_sut.reset_counters()

        self._setup_response_callback(is_offline=True)
        self._start_progress_thread()

        batch_size = self.batch_size
        num_samples = len(query_samples)

        logger.info(f"C++ Offline: {num_samples} samples, batch_size={batch_size}")

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
        self._setup_response_callback(is_offline=False)
        batch_size = self.batch_size

        for qs in query_samples:
            input_data = self._get_input_data(qs.index)

            # Handle batch dimension
            if batch_size == 1:
                if input_data.ndim == 3:
                    input_data = input_data[np.newaxis, ...]
            else:
                if input_data.ndim == 4 and input_data.shape[0] == 1:
                    input_data = input_data[0]
                if input_data.ndim == 3:
                    input_data = np.stack([input_data] * batch_size, axis=0)

            # Ensure float32 and contiguous
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            # Submit single sample (actual_batch_size=1)
            self._cpp_sut.start_async_batch(
                input_data,
                [qs.id],
                [qs.index],
                1
            )

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


def is_cpp_multi_die_sut_available() -> bool:
    """Check if C++ multi-die SUT is available."""
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
