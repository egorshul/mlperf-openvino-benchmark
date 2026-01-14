"""
MLPerf System Under Test (SUT) implementation for OpenVINO.
"""

import array
import logging
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class OpenVINOSUT:
    """
    System Under Test implementation using OpenVINO backend.
    
    This class implements the MLPerf LoadGen interface for inference testing.
    It handles:
    - Query dispatch
    - Asynchronous inference
    - Result collection
    - Performance metrics
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
        
        # For async processing
        self._query_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        
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
        Process queries in Server mode using async inference.

        In Server mode, queries arrive continuously and must be
        processed with low latency. We use AsyncInferQueue for parallelism.
        """
        # Start progress tracking if first query
        if self._sample_count == 0:
            self._start_progress(0, desc="Server inference (async)")

        num_samples = len(query_samples)

        # Prepare all inputs and metadata
        sample_data = []
        for qs in query_samples:
            sample_idx = qs.index
            features = self.qsl.get_features(sample_idx)
            inputs = {self.input_name: features.get("input", features.get(self.input_name))}
            sample_data.append({
                'query_id': qs.id,
                'sample_idx': sample_idx,
                'inputs': inputs,
            })

        # Results storage for this batch
        results = [None] * num_samples
        results_lock = threading.Lock()
        completed = [0]  # Use list for mutable counter in closure

        def on_complete(infer_request, userdata):
            """Callback when inference completes."""
            idx, query_id, sample_idx = userdata

            # Get output
            output = infer_request.get_output_tensor(0).data.copy()

            with results_lock:
                results[idx] = (query_id, sample_idx, output)
                completed[0] += 1

        # Create async queue with callback
        async_queue = self.backend.create_async_queue(callback=on_complete)

        # Start all inferences asynchronously
        for i, data in enumerate(sample_data):
            userdata = (i, data['query_id'], data['sample_idx'])
            self.backend.start_async_queue(async_queue, data['inputs'], userdata)

        # Wait for all to complete
        async_queue.wait_all()

        # Build responses
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        for query_id, sample_idx, result in results:
            # Store prediction
            self._predictions[sample_idx] = result

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
            self._sample_count += 1
            self._update_progress(1)

        # Report responses
        lg.QuerySamplesComplete(responses)

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
        # Close progress bar if still open (for Server mode)
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


class AsyncOpenVINOSUT(OpenVINOSUT):
    """
    Asynchronous SUT implementation for better throughput.
    
    Uses multiple inference requests and worker threads for
    maximum utilization of CPU resources.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        backend: OpenVINOBackend,
        qsl: QuerySampleLibrary,
        scenario: Scenario = Scenario.OFFLINE,
        num_workers: int = 0,
    ):
        """
        Initialize async SUT.
        
        Args:
            config: Benchmark configuration
            backend: OpenVINO backend instance
            qsl: Query Sample Library
            scenario: Test scenario
            num_workers: Number of worker threads (0 = auto)
        """
        super().__init__(config, backend, qsl, scenario)
        
        self.num_workers = num_workers if num_workers > 0 else backend.num_streams
        self._response_callbacks: Dict[int, Callable] = {}
    
    def _worker_thread(self, worker_id: int) -> None:
        """
        Worker thread for processing queries.
        
        Args:
            worker_id: Worker thread ID
        """
        logger.debug(f"Worker {worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Get work item
                work = self._query_queue.get(timeout=0.1)
                
                if work is None:
                    break
                
                query_id, sample_idx = work
                
                # Process sample
                _, result = self._process_sample(sample_idx)
                
                # Store prediction
                self._predictions[sample_idx] = result
                
                # Create and send response
                response_array = array.array('B', result.tobytes())
                bi = response_array.buffer_info()
                
                response = lg.QuerySampleResponse(
                    query_id,
                    bi[0],
                    bi[1]
                )
                
                lg.QuerySamplesComplete([response])
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def start_workers(self) -> None:
        """Start worker threads."""
        self._stop_event.clear()
        self._workers = []
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.num_workers} worker threads")
    
    def stop_workers(self) -> None:
        """Stop worker threads."""
        self._stop_event.set()
        
        # Send stop signals
        for _ in self._workers:
            self._query_queue.put(None)
        
        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self._workers.clear()
        logger.info("Workers stopped")
    
    def _issue_query_offline_async(
        self, 
        query_samples: List["lg.QuerySample"]
    ) -> None:
        """
        Process queries asynchronously in Offline mode.
        """
        # Queue all samples
        for qs in query_samples:
            self._query_queue.put((qs.id, qs.index))
        
        self._sample_count += len(query_samples)
        self._query_count += 1
    
    def issue_queries(self, query_samples: List["lg.QuerySample"]) -> None:
        """Process incoming queries."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline_async(query_samples)
        elif self.scenario == Scenario.SERVER:
            # For server mode, also use async processing
            for qs in query_samples:
                self._query_queue.put((qs.id, qs.index))
            self._sample_count += len(query_samples)
            self._query_count += 1
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")
    
    def flush_queries(self) -> None:
        """Wait for all pending queries to complete."""
        # Wait for queue to empty
        self._query_queue.join()
