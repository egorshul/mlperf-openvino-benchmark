"""
Multi-device SUT (System Under Test) for accelerators with multiple dies.

Optimized for maximum throughput on multi-die accelerators.
"""

import array
import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


class ResNetMultiDeviceSUT:
    """
    System Under Test for multi-die accelerator inference.

    Optimized for maximum throughput:
    - Proper batching support (batch_size > 1)
    - AsyncInferQueue per die for pipelining
    - Round-robin distribution across dies
    - Minimal Python overhead in hot path
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: MultiDeviceBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario
        self.batch_size = config.openvino.batch_size

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

        # Get input shape (after reshape) for batching
        self._input_shape = self.backend.input_shapes.get(self.input_name)

        # Results storage
        self._predictions: Dict[int, Any] = {}

        # Statistics
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0

        # Per-die async queues
        self._async_queues: Dict[str, AsyncInferQueue] = {}
        self._input_dtype = None
        self._issued_count = 0
        self._completed_count = 0
        self._die_index = 0
        self._die_lock = threading.Lock()

        # Offline mode: accumulated responses
        self._offline_responses: List[Tuple] = []  # (query_ids, response_arrays)
        self._offline_lock = threading.Lock()

        # Progress thread
        self._progress_thread_stop = False
        self._progress_thread = None

        # Setup async queues
        self._setup_async_queues()

        logger.info(
            f"ResNetMultiDeviceSUT: {self.backend.num_dies} dies, batch_size={self.batch_size}"
        )

    def _setup_async_queues(self) -> None:
        """Setup async inference queues for all dies."""
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

        # Create AsyncInferQueue for each die with high queue depth
        for die_name in self.backend.active_devices:
            ctx = self.backend._die_contexts[die_name]
            # High queue depth for maximum pipelining
            num_requests = max(ctx.optimal_nireq * 8, 32)

            async_queue = AsyncInferQueue(ctx.compiled_model, num_requests)
            async_queue.set_callback(self._make_callback(die_name))
            self._async_queues[die_name] = async_queue

            logger.info(f"AsyncInferQueue for {die_name}: {num_requests} requests")

    def _make_callback(self, die_name: str):
        """Create callback for async inference completion."""
        def callback(infer_request, userdata):
            query_ids, sample_indices, actual_batch_size, is_offline = userdata
            try:
                # Get batched output
                output = infer_request.get_output_tensor(0).data

                # Split batch output to individual samples
                for i in range(actual_batch_size):
                    sample_idx = sample_indices[i]
                    query_id = query_ids[i]

                    # Extract single sample from batch
                    if self.batch_size > 1:
                        sample_output = output[i].copy()
                    else:
                        sample_output = output.copy()

                    self._predictions[sample_idx] = sample_output

                    if is_offline:
                        # Offline: accumulate for batch response
                        response_array = array.array('B', sample_output.tobytes())
                        with self._offline_lock:
                            self._offline_responses.append((query_id, response_array))
                    else:
                        # Server: immediate response
                        response = lg.QuerySampleResponse(
                            query_id,
                            sample_output.ctypes.data,
                            sample_output.nbytes
                        )
                        lg.QuerySamplesComplete([response])

                self._completed_count += actual_batch_size

            except Exception as e:
                logger.error(f"Callback error on {die_name}: {e}")
                import traceback
                traceback.print_exc()

        return callback

    def _get_next_die(self) -> str:
        """Get next die for round-robin distribution (lock-free for speed)."""
        idx = self._die_index
        self._die_index = (idx + 1) % len(self.backend.active_devices)
        return self.backend.active_devices[idx]

    def _start_progress_thread(self):
        """Start progress monitoring thread."""
        self._progress_thread_stop = False
        self._start_time = time.time()

        def progress_fn():
            while not self._progress_thread_stop:
                time.sleep(0.3)
                elapsed = time.time() - self._start_time
                if elapsed > 0 and self._completed_count > 0:
                    throughput = self._completed_count / elapsed
                    pending = self._issued_count - self._completed_count
                    sys.stderr.write(
                        f"\r[{self.backend.num_dies} dies] "
                        f"done={self._completed_count}, pending={pending}, "
                        f"{throughput:.1f} samples/sec   "
                    )
                    sys.stderr.flush()

        self._progress_thread = threading.Thread(target=progress_fn, daemon=True)
        self._progress_thread.start()

    def _stop_progress_thread(self):
        """Stop progress thread and print final stats."""
        self._progress_thread_stop = True
        if self._progress_thread:
            self._progress_thread.join(timeout=1.0)

        elapsed = time.time() - self._start_time
        if elapsed > 0 and self._completed_count > 0:
            throughput = self._completed_count / elapsed
            sys.stderr.write(
                f"\nCompleted: {self._completed_count} samples in {elapsed:.2f}s "
                f"({throughput:.1f} samples/sec)\n"
            )
            sys.stderr.flush()

    def _issue_query_offline(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process queries in Offline mode with batching."""
        self._start_time = time.time()
        self._offline_responses.clear()
        self._issued_count = 0
        self._completed_count = 0

        qsl = self.qsl
        input_name = self.input_name
        input_dtype = self._input_dtype
        batch_size = self.batch_size
        num_samples = len(query_samples)

        logger.info(f"Offline: {num_samples} samples, batch_size={batch_size}, "
                   f"{self.backend.num_dies} dies")

        self._start_progress_thread()

        # Process in batches
        i = 0
        while i < num_samples:
            # Determine actual batch size (may be smaller at end)
            end_idx = min(i + batch_size, num_samples)
            actual_batch_size = end_idx - i

            # Collect batch data
            query_ids = []
            sample_indices = []
            batch_data = []

            for j in range(i, end_idx):
                qs = query_samples[j]
                sample_idx = qs.index
                query_id = qs.id

                query_ids.append(query_id)
                sample_indices.append(sample_idx)

                # Get input data
                if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                    input_data = qsl._loaded_samples[sample_idx]
                else:
                    features = qsl.get_features(sample_idx)
                    input_data = features.get("input", features.get(input_name))

                # Remove batch dimension if present (squeeze from [1,C,H,W] to [C,H,W])
                if input_data.ndim == 4 and input_data.shape[0] == 1:
                    input_data = input_data[0]

                batch_data.append(input_data)

            # Stack into batch tensor [batch_size, C, H, W]
            if actual_batch_size == batch_size:
                batched_input = np.stack(batch_data, axis=0)
            else:
                # Pad incomplete batch
                padded = batch_data + [batch_data[-1]] * (batch_size - actual_batch_size)
                batched_input = np.stack(padded, axis=0)

            # Convert dtype
            if input_dtype is not None and batched_input.dtype != input_dtype:
                batched_input = batched_input.astype(input_dtype, copy=False)

            # Ensure contiguous
            if not batched_input.flags['C_CONTIGUOUS']:
                batched_input = np.ascontiguousarray(batched_input)

            # Select die and submit
            die_name = self._get_next_die()
            async_queue = self._async_queues[die_name]

            userdata = (query_ids, sample_indices, actual_batch_size, True)
            async_queue.start_async({input_name: batched_input}, userdata=userdata)

            self._issued_count += actual_batch_size
            i = end_idx

        # Wait for all to complete
        for async_queue in self._async_queues.values():
            async_queue.wait_all()

        self._stop_progress_thread()

        # Send all responses
        responses = []
        for query_id, response_array in self._offline_responses:
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)
        self._query_count += 1
        self._sample_count = self._completed_count

    def _issue_query_server(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process queries in Server mode (low latency, batch=1 typically)."""
        qsl = self.qsl
        input_name = self.input_name
        input_dtype = self._input_dtype
        batch_size = self.batch_size

        # For Server mode, process immediately
        # If batch_size > 1, we still need to batch, but for latency we use batch=1
        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Ensure correct batch dimension
            if batch_size == 1:
                # Keep as-is or ensure [1, C, H, W]
                if input_data.ndim == 3:
                    input_data = input_data[np.newaxis, ...]
            else:
                # Batch > 1: pad single sample to full batch
                if input_data.ndim == 4 and input_data.shape[0] == 1:
                    input_data = input_data[0]
                if input_data.ndim == 3:
                    # Replicate to fill batch (only first result matters)
                    input_data = np.stack([input_data] * batch_size, axis=0)

            # Convert dtype
            if input_dtype is not None and input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype, copy=False)

            # Select die and submit
            die_name = self._get_next_die()
            async_queue = self._async_queues[die_name]

            # actual_batch_size=1 means only first output is used
            userdata = ([query_id], [sample_idx], 1, False)
            async_queue.start_async({input_name: input_data}, userdata=userdata)

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
        for async_queue in self._async_queues.values():
            async_queue.wait_all()

        self._stop_progress_thread()

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
        return f"MultiDevice-{self.backend.num_dies}dies"

    def get_predictions(self) -> Dict[int, Any]:
        return self._predictions

    def reset(self) -> None:
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._issued_count = 0
        self._completed_count = 0
        self._die_index = 0
        self._offline_responses.clear()
