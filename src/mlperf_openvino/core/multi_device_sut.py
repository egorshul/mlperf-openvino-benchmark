"""
Multi-device SUT (System Under Test) for X accelerator with multiple dies.

This SUT distributes inference workload across all available X dies
for maximum throughput in MLPerf benchmarks.
"""

import array
import logging
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    import openvino as ov
    from openvino import AsyncInferQueue
    OPENVINO_ASYNC_AVAILABLE = True
except ImportError:
    OPENVINO_ASYNC_AVAILABLE = False
    AsyncInferQueue = None

from ..backends.multi_device_backend import MultiDeviceBackend
from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class MultiDeviceSUT:
    """
    System Under Test for multi-die X accelerator inference.

    Distributes inference workload across all available X dies
    using AsyncInferQueue per die for optimal throughput.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: MultiDeviceBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize multi-device SUT.

        Args:
            config: Benchmark configuration
            backend: Multi-device backend instance
            qsl: Query Sample Library
            scenario: Test scenario
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario

        # Ensure backend is loaded
        if not self.backend.is_loaded:
            self.backend.load()

        # Get input/output names
        self.input_name = config.model.input_name
        if self.input_name not in self.backend.input_names:
            self.input_name = self.backend.input_names[0]

        self.output_name = config.model.output_name
        if self.output_name not in self.backend.output_names:
            self.output_name = self.backend.output_names[0]

        # Results storage
        self._predictions: Dict[int, Any] = {}

        # Statistics
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0
        self._end_time = 0.0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        # Per-die async queues for Server mode
        self._async_queues: Dict[str, AsyncInferQueue] = {}
        self._input_dtype = None
        self._issued_count = 0
        self._die_round_robin = 0
        self._die_lock = threading.Lock()

        if scenario == Scenario.SERVER:
            self._setup_async_server()

        logger.info(
            f"MultiDeviceSUT initialized with {self.backend.num_dies} dies: "
            f"{self.backend.active_devices}"
        )

    def _setup_async_server(self) -> None:
        """Setup async inference queues for all dies in Server mode."""
        self._start_time = time.time()
        self._issued_count = 0

        # Get input dtype from first die
        first_die = self.backend.active_devices[0]
        ctx = self.backend._die_contexts[first_die]
        compiled_model = ctx.compiled_model

        input_element_type = str(compiled_model.input(0).element_type)
        if 'f32' in input_element_type.lower():
            self._input_dtype = np.float32
        elif 'f16' in input_element_type.lower():
            self._input_dtype = np.float16
        elif 'i64' in input_element_type.lower():
            self._input_dtype = np.int64
        elif 'i32' in input_element_type.lower():
            self._input_dtype = np.int32
        else:
            self._input_dtype = np.float32

        # Create AsyncInferQueue for each die
        for die_name in self.backend.active_devices:
            ctx = self.backend._die_contexts[die_name]
            num_requests = max(ctx.optimal_nireq * 2, 8)

            async_queue = AsyncInferQueue(ctx.compiled_model, num_requests)

            # Create die-specific callback
            def make_callback(die):
                def callback(infer_request, userdata):
                    query_id, sample_idx = userdata
                    try:
                        output = infer_request.get_output_tensor(0).data
                        self._predictions[sample_idx] = output.copy()

                        response = lg.QuerySampleResponse(
                            query_id,
                            output.ctypes.data,
                            output.nbytes
                        )
                        lg.QuerySamplesComplete([response])
                    except Exception as e:
                        logger.error(f"Callback error on {die}: {e}")
                        response = lg.QuerySampleResponse(query_id, 0, 0)
                        lg.QuerySamplesComplete([response])

                    self._sample_count += 1
                return callback

            async_queue.set_callback(make_callback(die_name))
            self._async_queues[die_name] = async_queue

            logger.info(f"Created AsyncInferQueue for {die_name} with {num_requests} requests")

        # Start progress monitoring thread
        self._progress_thread_stop = False

        def progress_thread():
            while not self._progress_thread_stop:
                time.sleep(1.0)
                elapsed = time.time() - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                issued = self._issued_count
                pending = issued - self._sample_count
                dies_str = ",".join(self.backend.active_devices)
                print(
                    f"\rMulti-die [{dies_str}]: issued={issued}, done={self._sample_count}, "
                    f"pending={pending}, {throughput:.1f} samples/sec",
                    end="", flush=True
                )

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def _get_next_die(self) -> str:
        """Get next die for round-robin distribution."""
        with self._die_lock:
            die = self.backend.active_devices[self._die_round_robin % len(self.backend.active_devices)]
            self._die_round_robin += 1
        return die

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def _process_batch(self, sample_ids: List[int]) -> List[Tuple[int, np.ndarray]]:
        """Process a batch using multi-device backend."""
        batch_inputs = []
        for sample_id in sample_ids:
            features = self.qsl.get_features(sample_id)
            data = features.get("input", features.get(self.input_name))
            batch_inputs.append({self.input_name: data})

        batch_outputs = self.backend.predict_batch(batch_inputs)

        results = []
        for sample_id, output in zip(sample_ids, batch_outputs):
            result = output.get(self.output_name, list(output.values())[0])
            results.append((sample_id, result))

        return results

    def _issue_query_offline(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process queries in Offline mode using all dies."""
        responses = []
        response_arrays = []

        batch_size = self.config.openvino.batch_size if self.config.openvino.batch_size > 0 else 1

        sample_ids = [qs.id for qs in query_samples]
        sample_indices = [qs.index for qs in query_samples]

        total_samples = len(sample_indices)
        self._start_progress(total_samples, desc=f"Offline inference ({self.backend.num_dies} X dies)")

        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_ids = sample_ids[i:i + batch_size]

            batch_results = self._process_batch(batch_indices)

            for (idx, result), query_id in zip(batch_results, batch_ids):
                self._predictions[idx] = result

                response_array = array.array('B', result.tobytes())
                response_arrays.append(response_array)
                bi = response_array.buffer_info()

                response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
                responses.append(response)

            self._sample_count += len(batch_indices)
            self._update_progress(len(batch_indices))

        self._close_progress()
        lg.QuerySamplesComplete(responses)
        self._query_count += 1

    def _issue_query_server(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process queries in Server mode distributing across all dies."""
        qsl = self.qsl
        input_name = self.input_name
        input_dtype = self._input_dtype

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Convert dtype if needed
            if input_dtype is not None and input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype, copy=False)

            # Select die (round-robin)
            die_name = self._get_next_die()
            async_queue = self._async_queues[die_name]

            # Start async inference
            async_queue.start_async({input_name: input_data}, userdata=(query_id, sample_idx))
            self._issued_count += 1

        self._query_count += 1

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process incoming queries."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Wait for all async queues to complete
        for die_name, async_queue in self._async_queues.items():
            async_queue.wait_all()

        # Stop progress thread
        if hasattr(self, '_progress_thread_stop'):
            self._progress_thread_stop = True

        # Print final stats
        if self._async_queues:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            print(
                f"\nMulti-die completed: {self._sample_count} samples in {elapsed:.1f}s "
                f"({throughput:.1f} samples/sec) using {self.backend.num_dies} dies"
            )

        if self._progress_bar is not None:
            self._close_progress()

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
        return f"MultiDevice-X-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Any]:
        """Get all predictions."""
        return self._predictions

    def reset(self) -> None:
        """Reset SUT state."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._issued_count = 0
        self._die_round_robin = 0
