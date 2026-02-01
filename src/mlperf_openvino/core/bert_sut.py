"""BERT SUT implementation for MLPerf Question Answering."""

import array
import logging
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openvino import AsyncInferQueue

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

from .config import BenchmarkConfig, Scenario
from ..backends.base import BaseBackend
from ..datasets.squad import SQuADQSL

logger = logging.getLogger(__name__)


class BertSUT:
    """System Under Test for BERT Question Answering model."""

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: SQuADQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """Initialize BERT SUT."""
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

        self._map_input_names()

        self._predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._query_count = 0
        self._sample_count = 0

        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        self._sut = None
        self._qsl_handle = None

        self._async_queue = None
        self._issued_count = 0
        self._progress_stop_event = threading.Event()
        self._setup_async_server()

    def _setup_async_server(self) -> None:
        """Setup AsyncInferQueue for Server mode."""
        num_requests = self.backend.num_streams
        # Double for better pipelining
        num_requests = max(num_requests * 2, 16)

        compiled_model = self.backend._compiled_model
        self._async_queue = AsyncInferQueue(compiled_model, num_requests)

        def inference_callback(infer_request, userdata):
            """Callback when async inference completes."""
            query_id, sample_idx = userdata

            try:
                if self.start_logits_name == self.end_logits_name:
                    output = infer_request.get_output_tensor(0).data
                    if output.shape[-1] == 2:
                        start_logits = output[..., 0].copy()
                        end_logits = output[..., 1].copy()
                    else:
                        seq_len = output.shape[-1] // 2
                        start_logits = output[..., :seq_len].copy()
                        end_logits = output[..., seq_len:].copy()
                else:
                    start_logits = infer_request.get_tensor(self.start_logits_name).data.copy()
                    end_logits = infer_request.get_tensor(self.end_logits_name).data.copy()

                self._predictions[sample_idx] = (start_logits, end_logits)

                # array.array keeps response data alive through QuerySamplesComplete
                combined = np.stack([start_logits.flatten(), end_logits.flatten()], axis=-1)
                response_array = array.array('B', combined.astype(np.float32).tobytes())
                bi = response_array.buffer_info()

                response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

                self._sample_count += 1

            except Exception as e:
                logger.error(f"BERT callback error: {e}")
                response = lg.QuerySampleResponse(query_id, 0, 0)
                lg.QuerySamplesComplete([response])

        self._async_queue.set_callback(inference_callback)

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

    def _map_input_names(self) -> None:
        """Map expected input names to model's actual input names."""
        model_inputs = set(self.backend.input_names)

        self.input_ids_name = None
        self.attention_mask_name = None
        self.token_type_ids_name = None

        input_ids_patterns = ['input_ids', 'input_ids:0', 'input.1']
        attention_patterns = ['attention_mask', 'input_mask', 'attention_mask:0', 'input.2']
        token_type_patterns = ['token_type_ids', 'segment_ids', 'token_type_ids:0', 'input.3']

        for name in model_inputs:
            name_lower = name.lower()
            if any(p in name_lower for p in ['input_id', 'input.1']):
                self.input_ids_name = name
            elif any(p in name_lower for p in ['attention', 'mask', 'input.2']):
                self.attention_mask_name = name
            elif any(p in name_lower for p in ['token_type', 'segment', 'input.3']):
                self.token_type_ids_name = name

        if not all([self.input_ids_name, self.attention_mask_name, self.token_type_ids_name]):
            if len(self.backend.input_names) >= 3:
                self.input_ids_name = self.backend.input_names[0]
                self.attention_mask_name = self.backend.input_names[1]
                self.token_type_ids_name = self.backend.input_names[2]
            else:
                raise ValueError(
                    f"Could not map BERT inputs. Model inputs: {self.backend.input_names}"
                )

        self.start_logits_name = None
        self.end_logits_name = None

        model_outputs = self.backend.output_names
        if len(model_outputs) >= 2:
            for name in model_outputs:
                name_lower = name.lower()
                if 'start' in name_lower:
                    self.start_logits_name = name
                elif 'end' in name_lower:
                    self.end_logits_name = name

            if not self.start_logits_name:
                self.start_logits_name = model_outputs[0]
            if not self.end_logits_name:
                self.end_logits_name = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
        else:
            # Single output - assume concatenated
            self.start_logits_name = model_outputs[0]
            self.end_logits_name = model_outputs[0]

    def _process_sample(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single sample and return (start_logits, end_logits)."""
        features = self.qsl.get_features(sample_idx)

        # Ensure int64 dtype for BERT
        inputs = {
            self.input_ids_name: features['input_ids'].astype(np.int64),
            self.attention_mask_name: features['attention_mask'].astype(np.int64),
            self.token_type_ids_name: features['token_type_ids'].astype(np.int64),
        }

        outputs = self.backend.predict(inputs)

        if self.start_logits_name == self.end_logits_name:
            # Single output, need to split
            output = outputs[self.start_logits_name]
            if output.shape[-1] == 2:
                start_logits = output[..., 0]
                end_logits = output[..., 1]
            else:
                # Assume first half is start, second half is end
                seq_len = output.shape[-1] // 2
                start_logits = output[..., :seq_len]
                end_logits = output[..., seq_len:]
        else:
            start_logits = outputs[self.start_logits_name]
            end_logits = outputs[self.end_logits_name]

        return start_logits, end_logits

    def _process_batch(
        self,
        sample_ids: List[int]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Process a batch of samples."""
        results = []

        for sample_idx in sample_ids:
            result = self._process_sample(sample_idx)
            results.append(result)

        return results

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries in Offline mode using async inference for maximum throughput."""
        total_samples = len(query_samples)

        self._start_time = time.time()
        self._progress_stop_event.clear()
        self._issued_count = 0
        self._sample_count = 0

        def progress_thread():
            while not self._progress_stop_event.is_set():
                self._progress_stop_event.wait(timeout=1.0)
                if self._progress_stop_event.is_set():
                    break
                elapsed = time.time() - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                pending = self._issued_count - self._sample_count
                print(f"\rBERT Offline: issued={self._issued_count}, done={self._sample_count}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            features = self.qsl.get_features(sample_idx)

            inputs = {
                self.input_ids_name: features['input_ids'].astype(np.int64),
                self.attention_mask_name: features['attention_mask'].astype(np.int64),
                self.token_type_ids_name: features['token_type_ids'].astype(np.int64),
            }

            self._async_queue.start_async(inputs, userdata=(query_id, sample_idx))
            self._issued_count += 1

        self._async_queue.wait_all()

        self._progress_stop_event.set()

        elapsed = time.time() - self._start_time
        throughput = self._sample_count / elapsed if elapsed > 0 else 0
        print(f"\nBERT Offline completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

        self._query_count += 1

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries in Server mode using async inference."""
        if self._issued_count == 0:
            self._start_time = time.time()
            self._progress_stop_event.clear()
            self._start_server_progress_thread()

        for qs in query_samples:
            sample_idx = qs.index
            query_id = qs.id

            features = self.qsl.get_features(sample_idx)

            inputs = {
                self.input_ids_name: features['input_ids'].astype(np.int64),
                self.attention_mask_name: features['attention_mask'].astype(np.int64),
                self.token_type_ids_name: features['token_type_ids'].astype(np.int64),
            }

            self._async_queue.start_async(inputs, userdata=(query_id, sample_idx))
            self._issued_count += 1

        self._query_count += 1

    def _start_server_progress_thread(self) -> None:
        """Start progress monitoring thread for Server mode."""
        def progress_thread():
            while not self._progress_stop_event.is_set():
                self._progress_stop_event.wait(timeout=1.0)
                if self._progress_stop_event.is_set():
                    break
                elapsed = time.time() - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                pending = self._issued_count - self._sample_count
                print(f"\rBERT Server: issued={self._issued_count}, done={self._sample_count}, pending={pending}, {throughput:.1f} samples/sec", end="", flush=True)

        self._progress_thread = threading.Thread(target=progress_thread, daemon=True)
        self._progress_thread.start()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process incoming queries from LoadGen."""
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Offline mode already waits inside _issue_query_offline
        if self._async_queue is not None and self.scenario == Scenario.SERVER:
            self._async_queue.wait_all()

            self._progress_stop_event.set()

            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            print(f"\nBERT Server completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut is None:
            self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples,
            )
        return self._qsl_handle

    @property
    def name(self) -> str:
        """Get SUT name."""
        return f"BERT-{self.config.model.name}"

    def get_predictions(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset SUT state."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._issued_count = 0

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute F1 and exact match accuracy metrics."""
        if not self._predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        predictions = []
        indices = []

        for sample_idx, (start_logits, end_logits) in sorted(self._predictions.items()):
            indices.append(sample_idx)

        pred_texts = self.qsl.dataset.postprocess(
            [(self._predictions[idx][0], self._predictions[idx][1]) for idx in indices],
            indices
        )

        return self.qsl.dataset.compute_accuracy(pred_texts, indices)
