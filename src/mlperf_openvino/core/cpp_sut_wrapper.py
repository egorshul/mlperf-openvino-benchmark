"""
Wrapper for C++ SUT that integrates with the MLPerf LoadGen interface.

This module provides a Python wrapper around the C++ SUT extension,
handling the interface between LoadGen callbacks and C++ inference.
The actual inference runs in C++ without GIL for maximum throughput.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

# Try to import C++ extension
try:
    from ..cpp import CppSUT, CPP_AVAILABLE
except ImportError:
    CPP_AVAILABLE = False
    CppSUT = None

logger = logging.getLogger(__name__)


class CppSUTWrapper:
    """
    Wrapper for C++ SUT that provides MLPerf LoadGen interface.

    This class handles:
    - LoadGen integration (issue_queries, flush_queries)
    - QSL sample loading and caching
    - Response callback from C++ to LoadGen
    - Statistics and progress monitoring

    The actual inference runs in C++ without Python GIL for maximum throughput.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize the C++ SUT wrapper.

        Args:
            config: Benchmark configuration
            model_path: Path to ONNX or OpenVINO IR model
            qsl: Query Sample Library
            scenario: Test scenario (SERVER or OFFLINE)
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ SUT extension is not available. "
                "Build it with: python -m mlperf_openvino.cpp.setup_cpp"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create C++ SUT
        device = config.openvino.device
        num_streams = 0  # Auto
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        logger.info(f"Creating C++ SUT: device={device}, num_streams={num_streams}, hint={performance_hint}")
        self._cpp_sut = CppSUT(model_path, device, num_streams, performance_hint)

        # Load model
        logger.info("Loading model in C++ SUT...")
        self._cpp_sut.load()

        # Get input/output names
        self.input_name = self._cpp_sut.get_input_name()
        self.output_name = self._cpp_sut.get_output_name()
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")

        # Statistics
        self._start_time = 0.0
        self._progress_thread_stop = False

        # Set up response callback
        self._setup_response_callback()

        # Start progress monitoring
        self._start_progress_monitor()

        logger.info(f"C++ SUT initialized with {self._cpp_sut.get_optimal_nireq()} optimal requests")

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""

        def response_callback(query_id: int, output_data):
            """
            Called from C++ when inference completes.

            This runs in C++ thread but needs GIL for LoadGen call.
            The C++ binding handles GIL acquisition.
            """
            try:
                if output_data is not None:
                    # Create LoadGen response with output data
                    response = lg.QuerySampleResponse(
                        query_id,
                        output_data.ctypes.data,
                        output_data.nbytes
                    )
                else:
                    # Error case - empty response
                    response = lg.QuerySampleResponse(query_id, 0, 0)

                lg.QuerySamplesComplete([response])
            except Exception as e:
                logger.error(f"Response callback error: {e}")
                # Still try to complete to avoid hang
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def _start_progress_monitor(self):
        """Start progress monitoring thread."""
        self._start_time = time.time()
        self._progress_thread_stop = False

        def progress_thread():
            while not self._progress_thread_stop:
                time.sleep(1.0)
                elapsed = time.time() - self._start_time
                completed = self._cpp_sut.get_completed_count()
                issued = self._cpp_sut.get_issued_count()
                throughput = completed / elapsed if elapsed > 0 else 0
                pending = issued - completed
                print(f"\rC++ SUT: issued={issued}, done={completed}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process incoming queries using C++ SUT.

        This method is called by LoadGen when queries need to be processed.
        It prepares the input data and calls C++ start_async for each sample.

        Args:
            query_samples: List of query samples from LoadGen
        """
        qsl = self.qsl
        cpp_sut = self._cpp_sut
        input_name = self.input_name

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data from QSL
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Ensure float32 for C++ SUT
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)

            # Ensure contiguous array
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)

            # Start async inference in C++ (GIL released during inference!)
            cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        """
        Flush any pending queries.

        This is called by LoadGen when all queries have been issued.
        """
        # Wait for all C++ inferences to complete
        self._cpp_sut.wait_all()

        # Stop progress monitor
        self._progress_thread_stop = True

        # Print final stats
        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        throughput = completed / elapsed if elapsed > 0 else 0
        print(f"\nC++ SUT completed: {completed} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def get_sut(self) -> "lg.ConstructSUT":
        """Get the LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        """Get the LoadGen QSL object."""
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        """Get SUT name."""
        return f"OpenVINO-Cpp-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Any]:
        """Get all predictions (for accuracy mode)."""
        return self._cpp_sut.get_predictions()

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()


def create_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: QuerySampleLibrary,
    scenario: Scenario = Scenario.SERVER,
    force_python: bool = False,
):
    """
    Factory function to create the best available SUT.

    Uses C++ SUT for both Server and Offline modes when available.
    - Server: C++ SUT with THROUGHPUT hint (batch=1) for max parallelism
    - Offline: C++ SUT with THROUGHPUT hint (batch=N) for max throughput

    Args:
        config: Benchmark configuration
        model_path: Path to model file
        qsl: Query Sample Library
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        SUT instance (either CppSUTWrapper or OpenVINOSUT)
    """
    # Use C++ SUT for both Server and Offline modes if available
    if CPP_AVAILABLE and not force_python:
        batch_desc = "batch=1" if scenario == Scenario.SERVER else "batch=N"
        logger.info(f"Using C++ SUT for {scenario.value} mode (THROUGHPUT, {batch_desc})")
        return CppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python SUT
    logger.info("Using Python SUT (C++ extension not available)")
    from .sut import OpenVINOSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return OpenVINOSUT(config, backend, qsl, scenario)
