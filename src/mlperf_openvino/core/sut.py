"""
MLPerf System Under Test (SUT) implementation for OpenVINO.

Based on MLPerf reference implementation patterns:
- QueueRunner pattern for Server mode (decouple issue_queries from inference)
- Worker threads for parallel inference execution
- Minimal Python overhead in critical path
"""

import array
import logging
import sys
import time
import threading
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

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

from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.base import QuerySampleLibrary
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """Work item for the inference queue."""
    query_id: int
    sample_idx: int
    input_data: np.ndarray


class OpenVINOSUT:
    """
    System Under Test implementation using OpenVINO backend.

    This class implements the MLPerf LoadGen interface for inference testing.
    It uses the QueueRunner pattern from MLPerf reference implementation:
    - issue_queries() enqueues work items and returns immediately
    - Worker threads execute inference in parallel
    - Each worker calls QuerySamplesComplete after inference
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: OpenVINOBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize the SUT.

        Args:
            config: Benchmark configuration
            backend: OpenVINO backend instance
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

        # Get input name
        self.input_name = config.model.input_name
        if self.input_name not in self.backend.input_names:
            # Use first input if configured name not found
            self.input_name = self.backend.input_names[0]

        # Get output name
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
        self._progress_update_interval = 0.5  # seconds

        # Server mode: QueueRunner pattern (like MLPerf reference)
        self._task_queue: Optional[Queue] = None
        self._workers: List[threading.Thread] = []
        self._stop_workers = False

        if scenario == Scenario.SERVER:
            self._setup_queue_runner()

    def _setup_queue_runner(self) -> None:
        """
        Setup QueueRunner pattern for Server mode.

        Like MLPerf reference implementation:
        - Create task queue with sufficient buffer
        - Spawn worker threads (one per inference stream)
        - Each worker pulls from queue and executes inference
        """
        self._start_time = time.time()
        self._issued_count = 0
        self._stop_workers = False

        # Get optimal number of workers (= number of inference streams)
        num_workers = self.backend.num_streams
        queue_size = num_workers * 4  # Buffer for burst absorption

        logger.info(f"Server mode: QueueRunner with {num_workers} workers, queue_size={queue_size}")

        # Create task queue
        self._task_queue = Queue(maxsize=queue_size)

        # Pre-cache input dtype for fast conversion
        try:
            import openvino as ov
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
                self._input_dtype = None
        except Exception:
            self._input_dtype = None

        logger.info(f"Server mode: input dtype = {self._input_dtype}")

        # Create worker threads
        self._workers = []
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                daemon=True,
                name=f"InferWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {num_workers} inference worker threads")

        # Start progress monitoring thread
        self._progress_thread_stop = False
        def progress_thread():
            while not self._progress_thread_stop:
                time.sleep(1.0)
                elapsed = time.time() - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                issued = self._issued_count
                pending = issued - self._sample_count
                qsize = self._task_queue.qsize() if self._task_queue else 0
                print(f"\rServer: issued={issued}, done={self._sample_count}, pending={pending}, queue={qsize}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def _worker_thread(self, worker_id: int) -> None:
        """
        Worker thread that processes items from the queue.

        Each worker:
        1. Gets an item from the queue
        2. Runs inference using thread-safe predict
        3. Sends response via QuerySamplesComplete
        """
        input_name = self.input_name

        while not self._stop_workers:
            try:
                # Get work item with timeout (to check stop flag)
                item: WorkItem = self._task_queue.get(timeout=0.1)
            except:
                continue

            try:
                # Run inference using thread-safe predict
                inputs = {input_name: item.input_data}
                outputs = self.backend.predict_threadsafe(inputs)

                # Get output
                output = list(outputs.values())[0]

                # Store prediction
                self._predictions[item.sample_idx] = output

                # Send response immediately
                response = lg.QuerySampleResponse(
                    item.query_id,
                    output.ctypes.data,
                    output.nbytes
                )
                lg.QuerySamplesComplete([response])
                self._sample_count += 1

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                # Send empty response on error to avoid LoadGen hang
                response = lg.QuerySampleResponse(item.query_id, 0, 0)
                lg.QuerySamplesComplete([response])
                self._sample_count += 1
            finally:
                # Signal that task is done (for join() to work)
                self._task_queue.task_done()

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
            # Simple text-based progress update
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
        """
        Process a single sample.

        Args:
            sample_id: Sample index

        Returns:
            Tuple of (sample_id, result)
        """
        # Get input features
        features = self.qsl.get_features(sample_id)

        # Prepare input for backend
        inputs = {self.input_name: features.get("input", features.get(self.input_name))}

        # Run inference
        outputs = self.backend.predict(inputs)

        # Get output
        result = outputs.get(self.output_name, list(outputs.values())[0])

        return sample_id, result

    def _process_batch(
        self,
        sample_ids: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Process a batch of samples.

        Args:
            sample_ids: List of sample indices

        Returns:
            List of (sample_id, result) tuples
        """
        # Prepare batch input
        batch_inputs = []
        for sample_id in sample_ids:
            features = self.qsl.get_features(sample_id)
            data = features.get("input", features.get(self.input_name))
            batch_inputs.append({self.input_name: data})

        # Run batch inference
        batch_outputs = self.backend.predict_batch(batch_inputs)

        # Collect results
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
        logger.info(f"Offline: received {len(query_samples)} samples in issue_queries")

        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        # Process in batches for efficiency
        batch_size = self.config.openvino.batch_size if self.config.openvino.batch_size > 0 else 1

        sample_ids = [qs.id for qs in query_samples]
        sample_indices = [qs.index for qs in query_samples]

        # Start progress tracking
        total_samples = len(sample_indices)
        self._start_progress(total_samples, desc="Offline inference")

        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_ids = sample_ids[i:i + batch_size]

            # Process batch
            batch_results = self._process_batch(batch_indices)

            # Create responses
            for (idx, result), query_id in zip(batch_results, batch_ids):
                # Store prediction
                self._predictions[idx] = result

                # Create response
                response_array = array.array('B', result.tobytes())
                response_arrays.append(response_array)  # Keep alive!
                bi = response_array.buffer_info()

                response = lg.QuerySampleResponse(
                    query_id,
                    bi[0],
                    bi[1]
                )
                responses.append(response)

            # Update progress
            self._sample_count += len(batch_indices)
            self._update_progress(len(batch_indices))

        # Close progress
        self._close_progress()

        # Report all responses at once
        lg.QuerySamplesComplete(responses)

        self._query_count += 1

    def _issue_query_server(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process queries in Server mode using QueueRunner pattern.

        Like MLPerf reference implementation:
        - Enqueue work items and return immediately (non-blocking)
        - Worker threads handle inference and responses
        """
        # Cache references for faster access in loop
        qsl = self.qsl
        task_queue = self._task_queue
        input_name = self.input_name
        input_dtype = self._input_dtype

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            # Get input data directly from QSL cache
            if hasattr(qsl, '_loaded_samples') and sample_idx in qsl._loaded_samples:
                input_data = qsl._loaded_samples[sample_idx]
            else:
                features = qsl.get_features(sample_idx)
                input_data = features.get("input", features.get(input_name))

            # Convert dtype if needed
            if input_dtype is not None and input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype, copy=False)

            # Enqueue work item (non-blocking if queue has space)
            # put() will block only if queue is full, providing backpressure
            item = WorkItem(query_id=query_id, sample_idx=sample_idx, input_data=input_data)
            task_queue.put(item)
            self._issued_count += 1

        self._query_count += 1

    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """
        Process incoming queries.

        This method is called by LoadGen when queries need to be processed.

        Args:
            query_samples: List of query samples from LoadGen
        """
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """
        Flush any pending queries.

        This is called by LoadGen when all queries have been issued.
        """
        # Server mode: wait for all workers to complete
        if self._task_queue is not None:
            # Wait for queue to empty
            self._task_queue.join()

            # Stop workers
            self._stop_workers = True
            for worker in self._workers:
                worker.join(timeout=2.0)

            # Stop progress thread
            if hasattr(self, '_progress_thread_stop'):
                self._progress_thread_stop = True

            # Print final stats
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            print(f"\nServer completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

        # Close progress bar if still open
        if self._progress_bar is not None:
            self._close_progress()

    def get_sut(self) -> "lg.ConstructSUT":
        """
        Get the LoadGen SUT object.

        Returns:
            LoadGen SUT handle
        """
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self) -> "lg.ConstructQSL":
        """
        Get the LoadGen QSL object.

        Returns:
            LoadGen QSL handle
        """
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
