"""MLPerf System Under Test (SUT) implementation for OpenVINO."""

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

from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class OpenVINOSUT:
    """System Under Test implementation using OpenVINO backend."""

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: OpenVINOBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """Initialize the SUT."""
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario

        if not self.backend.is_loaded:
            self.backend.load()

        self.input_name = config.model.input_name
        if self.input_name not in self.backend.input_names:
            self.input_name = self.backend.input_names[0]

        self.output_name = config.model.output_name
        if self.output_name not in self.backend.output_names:
            self.output_name = self.backend.output_names[0]

        self._predictions: Dict[int, Any] = {}
        self._query_count = 0
        self._sample_count = 0
        self._start_time = 0.0
        self._end_time = 0.0

        self._progress_bar: Optional[Any] = None
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        self._async_queue: Optional[AsyncInferQueue] = None
        self._issued_count = 0
        self._input_dtype = None

        if scenario == Scenario.SERVER:
            self._setup_async_server()

    def _setup_async_server(self) -> None:
        """
        Setup optimized async inference for Server mode.

        Uses AsyncInferQueue directly (like Intel MLPerf submissions):
        - No Python worker threads (OpenVINO handles threading)
        - GIL released during inference
        - Immediate callback response
        """
        self._start_time = time.time()
        self._issued_count = 0

        num_requests = self.backend.num_streams
        num_requests = max(num_requests * 2, 16)

        compiled_model = self.backend._compiled_model
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
            self._input_dtype = np.float32  # Default

        self._async_queue = AsyncInferQueue(compiled_model, num_requests)

        # Define minimal callback - runs in OpenVINO thread (GIL acquired only for response)
        def inference_callback(infer_request, userdata):
            """Minimal callback: get output and send response immediately."""
            query_id, sample_idx = userdata

            try:
                # Get output tensor directly (minimal copy)
                output = infer_request.get_output_tensor(0).data

                self._predictions[sample_idx] = output.copy()

                # Send response immediately using output data pointer
                response = lg.QuerySampleResponse(
                    query_id,
                    output.ctypes.data,
                    output.nbytes
                )
                lg.QuerySamplesComplete([response])
            except Exception as e:
                logger.error(f"Callback error: {e}")
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

            self._sample_count += 1

        self._async_queue.set_callback(inference_callback)

        self._progress_thread_stop = False
        def progress_thread():
            while not self._progress_thread_stop:
                time.sleep(1.0)
                elapsed = time.time() - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                issued = self._issued_count
                pending = issued - self._sample_count
                print(f"\rServer: issued={issued}, done={self._sample_count}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

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

    def _process_sample(self, sample_id: int) -> Tuple[int, np.ndarray]:
        """Process a single sample."""
        features = self.qsl.get_features(sample_id)
        inputs = {self.input_name: features.get("input", features.get(self.input_name))}
        outputs = self.backend.predict(inputs)
        result = outputs.get(self.output_name, list(outputs.values())[0])
        return sample_id, result

    def _process_batch(
        self,
        sample_ids: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """Process a batch of samples."""
        batch_inputs = []
        for sample_id in sample_ids:
            features = self.qsl.get_features(sample_id)
            data = features.get("input", features.get(self.input_name))
            batch_inputs.append({self.input_name: data})

        batch_outputs = self.backend.predict_batch(batch_inputs)

        results = []
        for i, (sample_id, output) in enumerate(zip(sample_ids, batch_outputs)):
            result = output.get(self.output_name, list(output.values())[0])
            results.append((sample_id, result))

        return results

    def _issue_query_offline(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process queries in Offline mode.

        In Offline mode, all samples are sent at once and processed
        as fast as possible for maximum throughput.
        """
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        batch_size = self.config.openvino.batch_size if self.config.openvino.batch_size > 0 else 1

        sample_ids = [qs.id for qs in query_samples]
        sample_indices = [qs.index for qs in query_samples]

        total_samples = len(sample_indices)
        self._start_progress(total_samples, desc="Offline inference")

        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_ids = sample_ids[i:i + batch_size]

            batch_results = self._process_batch(batch_indices)

            for (idx, result), query_id in zip(batch_results, batch_ids):
                self._predictions[idx] = result

                response_array = array.array('B', result.tobytes())
                response_arrays.append(response_array)  # Keep alive!
                bi = response_array.buffer_info()

                response = lg.QuerySampleResponse(
                    query_id,
                    bi[0],
                    bi[1]
                )
                responses.append(response)

            self._sample_count += len(batch_indices)
            self._update_progress(len(batch_indices))

        self._close_progress()

        lg.QuerySamplesComplete(responses)

        self._query_count += 1

    def _issue_query_server(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process queries in Server mode using direct AsyncInferQueue.

        Optimized approach (like Intel/NVIDIA MLPerf submissions):
        - Direct start_async on AsyncInferQueue (non-blocking when slots available)
        - GIL released during inference wait
        - Callback handles response immediately
        """
        # Cache references for faster access in hot path
        qsl = self.qsl
        async_queue = self._async_queue
        input_name = self.input_name
        input_dtype = self._input_dtype

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data directly from QSL cache (fastest path)
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Convert dtype if needed (avoid copy if already correct)
            if input_dtype is not None and input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype, copy=False)

            # Start async inference directly on AsyncInferQueue
            # This is NON-BLOCKING when there's a free slot
            # When all slots busy, it waits (with GIL released!)
            async_queue.start_async({input_name: input_data}, userdata=(query_id, sample_idx))
            self._issued_count += 1

        self._query_count += 1

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process incoming queries from LoadGen."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._async_queue is not None:
            self._async_queue.wait_all()

            if hasattr(self, '_progress_thread_stop'):
                self._progress_thread_stop = True

            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            print(f"\nServer completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

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
        return f"OpenVINO-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Any]:
        """Get all predictions."""
        return self._predictions

    def reset(self) -> None:
        """Reset SUT state."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
