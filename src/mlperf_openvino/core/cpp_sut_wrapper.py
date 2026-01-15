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
    from ..cpp import (
        CppSUT, CppOfflineSUT, BertCppSUT, RetinaNetCppSUT,
        CPP_AVAILABLE, BERT_CPP_AVAILABLE, RETINANET_CPP_AVAILABLE
    )
except ImportError:
    CPP_AVAILABLE = False
    BERT_CPP_AVAILABLE = False
    RETINANET_CPP_AVAILABLE = False
    CppSUT = None
    CppOfflineSUT = None
    BertCppSUT = None
    RetinaNetCppSUT = None

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
        self._progress_stop_event = threading.Event()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

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
        self._progress_stop_event.clear()

        def progress_thread():
            while not self._progress_stop_event.is_set():
                self._progress_stop_event.wait(timeout=1.0)
                if self._progress_stop_event.is_set():
                    break
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
        self._progress_stop_event.set()

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

    @property
    def _sample_count(self) -> int:
        """Get number of completed samples (for compatibility with benchmark_runner)."""
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        """Get number of issued queries (for compatibility with benchmark_runner)."""
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Any]:
        """Get all predictions (for accuracy mode)."""
        # C++ returns Dict[int, List[float]], convert to numpy arrays
        raw_predictions = self._cpp_sut.get_predictions()
        return {idx: np.array(pred, dtype=np.float32) for idx, pred in raw_predictions.items()}

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()


class CppOfflineSUTWrapper:
    """
    Wrapper for C++ Offline SUT with batch inference.

    Optimized for Offline scenario:
    - Sync batch inference (multiple samples per call)
    - No per-sample callback overhead
    - Maximum throughput with large batches
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: QuerySampleLibrary,
        batch_size: int = 32,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not CPP_AVAILABLE or CppOfflineSUT is None:
            raise ImportError("C++ Offline SUT extension not available")

        self.config = config
        self.qsl = qsl
        self.batch_size = batch_size
        self.scenario = Scenario.OFFLINE

        # Create C++ Offline SUT
        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        logger.info(f"Creating C++ Offline SUT: device={device}, batch_size={batch_size}")
        self._cpp_sut = CppOfflineSUT(model_path, device, batch_size, num_streams)

        # Load model
        logger.info("Loading model in C++ Offline SUT...")
        self._cpp_sut.load()

        # Get input info
        self.input_name = self._cpp_sut.get_input_name()
        self.output_name = self._cpp_sut.get_output_name()
        self.sample_size = self._cpp_sut.get_sample_size()

        logger.info(f"C++ Offline SUT initialized: batch={batch_size}, sample_size={self.sample_size}")

        # Statistics
        self._start_time = 0.0
        self._issued_count = 0

        # Predictions storage for accuracy mode
        self._predictions: Dict[int, Any] = {}

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process queries using batch inference.

        For Offline mode, all samples come at once - process in batches.
        """
        self._start_time = time.time()
        total_samples = len(query_samples)
        self._issued_count = total_samples

        logger.info(f"Offline: Processing {total_samples} samples in batches of {self.batch_size}")

        # Process in batches
        for batch_start in range(0, total_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_samples)
            batch_samples = query_samples[batch_start:batch_end]
            num_in_batch = len(batch_samples)

            # Prepare batch input
            batch_input = np.zeros((self.batch_size, self.sample_size), dtype=np.float32)

            query_ids = []
            for i, qs in enumerate(batch_samples):
                sample_idx = qs.index
                query_ids.append(qs.id)

                # Get input data from QSL
                if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
                    input_data = self.qsl._loaded_samples[sample_idx]
                else:
                    features = self.qsl.get_features(sample_idx)
                    input_data = features.get("input", features.get(self.input_name))

                # Flatten and copy to batch
                batch_input[i] = input_data.flatten().astype(np.float32)

            # Run batch inference
            results = self._cpp_sut.infer_batch(batch_input.flatten(), num_in_batch)

            # Send responses to LoadGen and store predictions
            responses = []
            for i, (query_id, result) in enumerate(zip(query_ids, results)):
                sample_idx = batch_samples[i].index

                # Store prediction for accuracy computation
                self._predictions[sample_idx] = result.copy()

                response = lg.QuerySampleResponse(
                    query_id,
                    result.ctypes.data,
                    result.nbytes
                )
                responses.append(response)

            lg.QuerySamplesComplete(responses)

            # Progress
            completed = batch_end
            elapsed = time.time() - self._start_time
            throughput = completed / elapsed if elapsed > 0 else 0
            print(f"\rOffline: {completed}/{total_samples} samples, {throughput:.1f} samples/sec", end="", flush=True)

        print()  # Newline after progress

    def flush_queries(self) -> None:
        """Flush completed - all done in issue_queries for Offline."""
        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        throughput = completed / elapsed if elapsed > 0 else 0
        logger.info(f"Offline completed: {completed} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

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
        return f"OpenVINO-CppOffline-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._issued_count

    def get_predictions(self) -> Dict[int, Any]:
        """Get stored predictions for accuracy computation."""
        return self._predictions

    def reset(self) -> None:
        self._cpp_sut.reset_counters()
        self._issued_count = 0
        self._predictions = {}


class BertCppSUTWrapper:
    """
    Wrapper for BERT C++ SUT with 3 int64 inputs and 2 float32 outputs.

    Optimized for BERT Question Answering:
    - input_ids, attention_mask, token_type_ids (int64)
    - start_logits, end_logits (float32)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: "QuerySampleLibrary",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not BERT_CPP_AVAILABLE or BertCppSUT is None:
            raise ImportError(
                "BERT C++ SUT extension not available. "
                "Build it with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create BERT C++ SUT
        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        logger.info(f"Creating BERT C++ SUT: device={device}, num_streams={num_streams}, hint={performance_hint}")
        self._cpp_sut = BertCppSUT(model_path, device, num_streams, performance_hint)

        # Load model
        logger.info("Loading model in BERT C++ SUT...")
        self._cpp_sut.load()

        # Get input/output names
        self.input_ids_name = self._cpp_sut.get_input_ids_name()
        self.attention_mask_name = self._cpp_sut.get_attention_mask_name()
        self.token_type_ids_name = self._cpp_sut.get_token_type_ids_name()
        self.start_logits_name = self._cpp_sut.get_start_logits_name()
        self.end_logits_name = self._cpp_sut.get_end_logits_name()
        self.seq_length = self._cpp_sut.get_seq_length()

        logger.info(f"BERT inputs: {self.input_ids_name}, {self.attention_mask_name}, {self.token_type_ids_name}")
        logger.info(f"BERT outputs: {self.start_logits_name}, {self.end_logits_name}")
        logger.info(f"Sequence length: {self.seq_length}")

        # Statistics
        self._start_time = 0.0
        self._progress_stop_event = threading.Event()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Set up response callback
        self._setup_response_callback()

        # Start progress monitoring
        self._start_progress_monitor()

        logger.info(f"BERT C++ SUT initialized with {self._cpp_sut.get_optimal_nireq()} optimal requests")

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""
        import array

        # Pre-allocate a single dummy response for performance mode
        # LoadGen only cares that we called QuerySamplesComplete
        # The actual data is only needed for accuracy mode (stored in C++ predictions)
        dummy_data = array.array('B', [0] * 8)  # Minimal response
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, start_logits, end_logits):
            # Use minimal response - predictions are stored in C++ for accuracy mode
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def _start_progress_monitor(self):
        """Start progress monitoring thread."""
        self._start_time = time.time()
        self._progress_stop_event.clear()

        def progress_thread():
            while not self._progress_stop_event.is_set():
                self._progress_stop_event.wait(timeout=1.0)
                if self._progress_stop_event.is_set():
                    break
                elapsed = time.time() - self._start_time
                completed = self._cpp_sut.get_completed_count()
                issued = self._cpp_sut.get_issued_count()
                throughput = completed / elapsed if elapsed > 0 else 0
                pending = issued - completed
                print(f"\rBERT C++ SUT: issued={issued}, done={completed}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries using BERT C++ SUT."""
        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get pre-optimized features from QSL (already int64, contiguous, flattened)
            features = self.qsl.get_features(sample_idx)

            # Features are pre-converted to int64 contiguous arrays in QSL
            # Just pass them directly to C++ SUT
            self._cpp_sut.start_async(
                features['input_ids'],
                features['attention_mask'],
                features['token_type_ids'],
                query_id,
                sample_idx
            )

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Wait for all C++ inferences to complete
        self._cpp_sut.wait_all()

        # Stop progress monitor
        self._progress_stop_event.set()

        # Print final stats
        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        throughput = completed / elapsed if elapsed > 0 else 0
        print(f"\nBERT C++ SUT completed: {completed} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

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
        return f"OpenVINO-BertCpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, tuple]:
        """Get all predictions as dict of {sample_idx: (start_logits, end_logits)}."""
        raw_preds = self._cpp_sut.get_predictions()
        # Convert from dict with 'start_logits' and 'end_logits' keys to tuple
        result = {}
        for idx, pred in raw_preds.items():
            start_logits = np.array(pred['start_logits'], dtype=np.float32)
            end_logits = np.array(pred['end_logits'], dtype=np.float32)
            result[idx] = (start_logits, end_logits)
        return result

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute BERT accuracy metrics (F1 and Exact Match).

        Returns:
            Dictionary with f1, exact_match, and num_samples
        """
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        # Extract indices in sorted order
        indices = sorted(predictions.keys())

        # Use dataset's postprocess to get answer texts
        pred_texts = self.qsl.dataset.postprocess(
            [(predictions[idx][0], predictions[idx][1]) for idx in indices],
            indices
        )

        # Compute F1 and EM using dataset's method
        return self.qsl.dataset.compute_accuracy(pred_texts, indices)


def create_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: QuerySampleLibrary,
    scenario: Scenario = Scenario.SERVER,
    force_python: bool = False,
):
    """
    Factory function to create the best available SUT.

    - Server: CppSUT with async inference (batch=1)
    - Offline: CppOfflineSUT with sync batch inference

    Args:
        config: Benchmark configuration
        model_path: Path to model file
        qsl: Query Sample Library
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        SUT instance
    """
    if CPP_AVAILABLE and not force_python:
        # Use async C++ SUT for both modes - it has parallelism via InferRequest pool
        # CppOfflineSUT (sync batch) is slower because it uses single InferRequest
        mode_desc = "async" if scenario == Scenario.SERVER else "async-parallel"
        logger.info(f"Using C++ SUT for {scenario.value} mode ({mode_desc})")
        return CppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python SUT
    logger.info(f"Using Python SUT for {scenario.value} mode")
    from .sut import OpenVINOSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return OpenVINOSUT(config, backend, qsl, scenario)


def create_bert_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
    force_python: bool = False,
):
    """
    Factory function to create BERT SUT.

    Uses C++ SUT if available for maximum performance.

    Args:
        config: Benchmark configuration
        model_path: Path to BERT model file
        qsl: Query Sample Library (SQuADQSL)
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        BERT SUT instance
    """
    if BERT_CPP_AVAILABLE and not force_python:
        logger.info(f"Using BERT C++ SUT for {scenario.value} mode")
        return BertCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python BertSUT
    logger.info(f"Using Python BERT SUT for {scenario.value} mode")
    from .bert_sut import BertSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return BertSUT(config, backend, qsl, scenario)


class RetinaNetCppSUTWrapper:
    """
    Wrapper for RetinaNet C++ SUT with object detection outputs.

    Optimized for RetinaNet Object Detection:
    - 1 input: float32 image
    - 3 outputs: boxes, scores, labels
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        qsl: "QuerySampleLibrary",
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not RETINANET_CPP_AVAILABLE or RetinaNetCppSUT is None:
            raise ImportError(
                "RetinaNet C++ SUT extension not available. "
                "Build it with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Create RetinaNet C++ SUT
        device = config.openvino.device
        num_streams = 0
        if config.openvino.num_streams != "AUTO":
            try:
                num_streams = int(config.openvino.num_streams)
            except ValueError:
                pass

        performance_hint = config.openvino.performance_hint

        logger.info(f"Creating RetinaNet C++ SUT: device={device}, hint={performance_hint}")
        self._cpp_sut = RetinaNetCppSUT(model_path, device, num_streams, performance_hint)

        # Load model
        logger.info("Loading model in RetinaNet C++ SUT...")
        self._cpp_sut.load()

        # Get info
        self.input_name = self._cpp_sut.get_input_name()
        self.boxes_name = self._cpp_sut.get_boxes_name()
        self.scores_name = self._cpp_sut.get_scores_name()
        self.labels_name = self._cpp_sut.get_labels_name()

        logger.info(f"RetinaNet input: {self.input_name}")
        logger.info(f"RetinaNet outputs: boxes={self.boxes_name}, scores={self.scores_name}, labels={self.labels_name}")

        # Statistics
        self._start_time = 0.0
        self._progress_stop_event = threading.Event()

        # Enable storing predictions for accuracy mode
        self._cpp_sut.set_store_predictions(True)

        # Set up response callback
        self._setup_response_callback()

        # Start progress monitoring
        self._start_progress_monitor()

        logger.info(f"RetinaNet C++ SUT initialized with {self._cpp_sut.get_optimal_nireq()} optimal requests")

    def _setup_response_callback(self):
        """Set up the callback that C++ uses to notify LoadGen."""
        import array

        # Use minimal response for performance mode
        dummy_data = array.array('B', [0] * 8)
        dummy_bi = dummy_data.buffer_info()

        def response_callback(query_id: int, boxes, scores, labels):
            # Use minimal response - predictions are stored in C++ for accuracy mode
            response = lg.QuerySampleResponse(query_id, dummy_bi[0], dummy_bi[1])
            lg.QuerySamplesComplete([response])

        self._cpp_sut.set_response_callback(response_callback)

    def _start_progress_monitor(self):
        """Start progress monitoring thread."""
        self._start_time = time.time()
        self._progress_stop_event.clear()

        def progress_thread():
            while not self._progress_stop_event.is_set():
                self._progress_stop_event.wait(timeout=1.0)
                if self._progress_stop_event.is_set():
                    break
                elapsed = time.time() - self._start_time
                completed = self._cpp_sut.get_completed_count()
                issued = self._cpp_sut.get_issued_count()
                throughput = completed / elapsed if elapsed > 0 else 0
                pending = issued - completed
                print(f"\rRetinaNet C++ SUT: issued={issued}, done={completed}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries using RetinaNet C++ SUT."""
        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get pre-optimized features from QSL
            features = self.qsl.get_features(sample_idx)
            input_data = features.get('input', features.get(self.input_name))

            # Flatten and ensure float32 contiguous
            input_data = np.ascontiguousarray(input_data.flatten(), dtype=np.float32)

            # Start async inference in C++
            self._cpp_sut.start_async(input_data, query_id, sample_idx)

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        self._cpp_sut.wait_all()

        # Stop progress monitor
        self._progress_stop_event.set()

        # Print final stats
        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        throughput = completed / elapsed if elapsed > 0 else 0
        print(f"\nRetinaNet C++ SUT completed: {completed} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

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
        return f"OpenVINO-RetinaNetCpp-{self.config.model.name}"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    @property
    def _query_count(self) -> int:
        return self._cpp_sut.get_issued_count()

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get all predictions as dict of {sample_idx: {boxes, scores, labels}}."""
        raw_preds = self._cpp_sut.get_predictions()
        result = {}
        for idx, pred in raw_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'].size > 0 else np.array([])

            # Reshape boxes to [N, 4]
            if len(boxes) > 0 and len(boxes) % 4 == 0:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels.astype(np.int64) if len(labels) > 0 else np.array([], dtype=np.int64),
            }
        return result

    def reset(self) -> None:
        """Reset SUT state."""
        self._cpp_sut.reset_counters()

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute mAP accuracy."""
        predictions = self.get_predictions()

        if not predictions:
            return {'mAP': 0.0, 'num_samples': 0}

        pred_list = []
        gt_list = []
        indices = sorted(predictions.keys())

        for idx in indices:
            pred_list.append(predictions[idx])
            gt_list.append(self.qsl.get_label(idx))

        return self.qsl.dataset.compute_accuracy(pred_list, indices)


def create_retinanet_sut(
    config: BenchmarkConfig,
    model_path: str,
    qsl: "QuerySampleLibrary",
    scenario: Scenario = Scenario.OFFLINE,
    force_python: bool = False,
):
    """
    Factory function to create RetinaNet SUT.

    Uses C++ SUT if available for maximum performance.

    Args:
        config: Benchmark configuration
        model_path: Path to RetinaNet model file
        qsl: Query Sample Library (OpenImagesQSL)
        scenario: Test scenario
        force_python: Force Python SUT even if C++ is available

    Returns:
        RetinaNet SUT instance
    """
    if RETINANET_CPP_AVAILABLE and not force_python:
        logger.info(f"Using RetinaNet C++ SUT for {scenario.value} mode")
        return RetinaNetCppSUTWrapper(config, model_path, qsl, scenario)

    # Fall back to Python RetinaNetSUT
    logger.info(f"Using Python RetinaNet SUT for {scenario.value} mode")
    from .retinanet_sut import RetinaNetSUT
    from ..backends.openvino_backend import OpenVINOBackend

    backend = OpenVINOBackend(model_path, config.openvino)
    return RetinaNetSUT(config, backend, qsl, scenario)
