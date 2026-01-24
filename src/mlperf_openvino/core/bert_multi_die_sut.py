"""
Python wrapper for C++ BertMultiDieCppSUT.

Clean implementation for BERT on multi-die accelerators.
Handles 3 int64 inputs (input_ids, attention_mask, token_type_ids)
and 2 float32 outputs (start_logits, end_logits).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    from ..cpp import BertMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and BertMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    BertMultiDieCppSUT = None

from ..datasets.squad import SQuADQSL
from ..core.config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)


class BertMultiDieCppSUTWrapper:
    """
    Wrapper for C++ BERT multi-die SUT with MLPerf LoadGen integration.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: SQuADQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen not installed")

        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ BERT multi-die SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Batch size: 1 for Server, configured for Offline
        if scenario == Scenario.SERVER:
            self.batch_size = 1
        else:
            self.batch_size = config.openvino.batch_size

        # Build compile properties
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = dict(config.openvino.device_properties)

        # Get nireq_multiplier from config
        server_config = config.model.server if hasattr(config.model, 'server') else None
        if scenario == Scenario.SERVER:
            nireq_multiplier = getattr(server_config, 'nireq_multiplier', 2) if server_config else 2
        else:
            nireq_multiplier = 4

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = BertMultiDieCppSUT(
            config.model.model_path,
            device_prefix,
            self.batch_size,
            compile_props,
            nireq_multiplier
        )

        # Set specific target devices if a specific die is selected (e.g., NPU.0)
        if config.openvino.is_specific_die():
            target_devices = [config.openvino.device]
            self._cpp_sut.set_target_devices(target_devices)
            logger.debug(f"Set target devices: {target_devices}")

        # Statistics
        self._start_time = 0.0
        self._query_count = 0
        self._server_callback_set = False
        self._qsl_validated = False
        self._is_accuracy_mode = False

        logger.debug(f"BertMultiDieCppSUTWrapper: device_prefix={device_prefix}, batch_size={self.batch_size}")

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load and compile the model."""
        # Configure explicit batching if enabled (before load)
        if self.scenario == Scenario.SERVER:
            server_config = self.config.model.server if hasattr(self.config.model, 'server') else None
            if server_config and getattr(server_config, 'explicit_batching', False):
                batch_size = getattr(server_config, 'explicit_batch_size', 4)
                timeout_us = getattr(server_config, 'batch_timeout_us', 500)
                self._cpp_sut.enable_explicit_batching(True, batch_size, timeout_us)

        self._cpp_sut.load()
        self._cpp_sut.set_store_predictions(is_accuracy_mode)
        self._is_accuracy_mode = is_accuracy_mode

        logger.debug(
            f"BERT C++ SUT loaded: {self._cpp_sut.get_num_dies()} dies, "
            f"{self._cpp_sut.get_total_requests()} total requests, "
            f"seq_len={self._cpp_sut.get_seq_length()}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._cpp_sut.is_loaded()

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _prevalidate_qsl_data(self) -> None:
        """Register QSL data pointers in C++ for fast Server mode dispatch."""
        self._qsl_data_ready = False
        self._cpp_qsl_registered = False

        if not hasattr(self.qsl, '_loaded_samples'):
            logger.warning("QSL does not have _loaded_samples")
            return

        if not self.qsl._loaded_samples:
            return

        try:
            registered_count = 0
            for idx, data in self.qsl._loaded_samples.items():
                # BERT samples have input_ids, attention_mask, token_type_ids
                input_ids = data.get('input_ids')
                attention_mask = data.get('attention_mask')
                token_type_ids = data.get('token_type_ids')

                if input_ids is None or attention_mask is None or token_type_ids is None:
                    continue

                # Ensure contiguous int64 arrays
                if not input_ids.flags['C_CONTIGUOUS']:
                    input_ids = np.ascontiguousarray(input_ids)
                    self.qsl._loaded_samples[idx]['input_ids'] = input_ids
                if not attention_mask.flags['C_CONTIGUOUS']:
                    attention_mask = np.ascontiguousarray(attention_mask)
                    self.qsl._loaded_samples[idx]['attention_mask'] = attention_mask
                if not token_type_ids.flags['C_CONTIGUOUS']:
                    token_type_ids = np.ascontiguousarray(token_type_ids)
                    self.qsl._loaded_samples[idx]['token_type_ids'] = token_type_ids

                self._cpp_sut.register_sample_data(
                    idx, input_ids, attention_mask, token_type_ids
                )
                registered_count += 1

            self._cpp_qsl_registered = True
            self._qsl_data_ready = True
            logger.debug(f"Registered {registered_count} BERT samples in C++")
        except Exception as e:
            logger.debug(f"Failed to register BERT samples: {e}")
            self._cpp_qsl_registered = False

    def _get_input_data(self, sample_idx: int) -> Dict[str, np.ndarray]:
        """Get input data for a sample."""
        if hasattr(self.qsl, '_loaded_samples') and sample_idx in self.qsl._loaded_samples:
            return self.qsl._loaded_samples[sample_idx]
        else:
            return self.qsl.get_features(sample_idx)

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode with batching."""
        self._start_time = time.time()
        self._cpp_sut.reset_counters()
        self._cpp_sut.enable_direct_loadgen(True)

        batch_size = self.batch_size
        num_samples = len(query_samples)
        seq_length = self._cpp_sut.get_seq_length()

        i = 0
        while i < num_samples:
            end_idx = min(i + batch_size, num_samples)
            actual_batch_size = end_idx - i

            query_ids = []
            sample_indices = []
            batch_input_ids = []
            batch_attention_mask = []
            batch_token_type_ids = []

            for j in range(i, end_idx):
                qs = query_samples[j]
                query_ids.append(qs.id)
                sample_indices.append(qs.index)

                data = self._get_input_data(qs.index)
                batch_input_ids.append(data['input_ids'].flatten())
                batch_attention_mask.append(data['attention_mask'].flatten())
                batch_token_type_ids.append(data['token_type_ids'].flatten())

            # Pad batch if needed
            if actual_batch_size < batch_size:
                for _ in range(batch_size - actual_batch_size):
                    batch_input_ids.append(batch_input_ids[-1])
                    batch_attention_mask.append(batch_attention_mask[-1])
                    batch_token_type_ids.append(batch_token_type_ids[-1])

            input_ids = np.stack(batch_input_ids, axis=0).astype(np.int64)
            attention_mask = np.stack(batch_attention_mask, axis=0).astype(np.int64)
            token_type_ids = np.stack(batch_token_type_ids, axis=0).astype(np.int64)

            if not input_ids.flags['C_CONTIGUOUS']:
                input_ids = np.ascontiguousarray(input_ids)
            if not attention_mask.flags['C_CONTIGUOUS']:
                attention_mask = np.ascontiguousarray(attention_mask)
            if not token_type_ids.flags['C_CONTIGUOUS']:
                token_type_ids = np.ascontiguousarray(token_type_ids)

            self._cpp_sut.start_async_batch(
                input_ids, attention_mask, token_type_ids,
                query_ids, sample_indices, actual_batch_size
            )
            i = end_idx

        self._cpp_sut.wait_all()
        self._query_count += 1

    def _issue_query_server(self, query_samples: List) -> None:
        """Process queries in Server mode."""
        if not self._server_callback_set:
            self._cpp_sut.enable_direct_loadgen(True)
            self._server_callback_set = True

        if not self._qsl_validated:
            self._prevalidate_qsl_data()
            self._qsl_validated = True

        if self._cpp_qsl_registered:
            query_ids = [qs.id for qs in query_samples]
            sample_indices = [qs.index for qs in query_samples]
            self._cpp_sut.issue_queries_server_fast(query_ids, sample_indices)
        else:
            # Fallback path
            for qs in query_samples:
                data = self._get_input_data(qs.index)
                input_ids = np.ascontiguousarray(data['input_ids'].flatten(), dtype=np.int64)
                attention_mask = np.ascontiguousarray(data['attention_mask'].flatten(), dtype=np.int64)
                token_type_ids = np.ascontiguousarray(data['token_type_ids'].flatten(), dtype=np.int64)

                # Add batch dimension
                input_ids = input_ids.reshape(1, -1)
                attention_mask = attention_mask.reshape(1, -1)
                token_type_ids = token_type_ids.reshape(1, -1)

                self._cpp_sut.start_async_batch(
                    input_ids, attention_mask, token_type_ids,
                    [qs.id], [qs.index], 1
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

    def get_sut(self):
        """Get LoadGen SUT object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def supports_native_benchmark(self) -> bool:
        """Check if native C++ benchmark is supported."""
        return self.scenario == Scenario.SERVER

    def run_native_benchmark(
        self,
        test_settings: "lg.TestSettings",
        log_settings: "lg.LogSettings",
    ) -> None:
        """Run benchmark with pure C++ SUT."""
        if not self.supports_native_benchmark():
            raise RuntimeError("Native benchmark not supported for this scenario")

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Load samples
        if self._is_accuracy_mode:
            num_samples = self.qsl.total_sample_count
        else:
            num_samples = min(self.qsl.performance_sample_count, self.qsl.total_sample_count)

        sample_indices = list(range(num_samples))
        self.qsl.load_query_samples(sample_indices)

        # Register samples in C++
        self._prevalidate_qsl_data()

        if not self._cpp_qsl_registered:
            raise RuntimeError("Failed to register samples in C++")

        total_count = self.qsl.total_sample_count
        perf_count = self.qsl.performance_sample_count

        target_qps = getattr(test_settings, 'server_target_qps', 0.0)
        target_latency_ns = getattr(test_settings, 'server_target_latency_ns', 0)
        min_duration_ms = getattr(test_settings, 'min_duration_ms', 0)
        min_query_count = getattr(test_settings, 'min_query_count', 0)

        self._cpp_sut.reset_counters()

        log_output_dir = getattr(log_settings.log_output, 'outdir', '.')

        self._cpp_sut.run_server_benchmark(
            total_count,
            perf_count,
            "",
            "",
            log_output_dir,
            target_qps,
            target_latency_ns,
            min_duration_ms,
            min_query_count,
            self._is_accuracy_mode
        )

        self._query_count = self._cpp_sut.get_issued_count()

        self.qsl.unload_query_samples(sample_indices)

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
        return f"BertMultiDieCpp-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get stored predictions as dict of {sample_idx: (start_logits, end_logits)}."""
        cpp_preds = self._cpp_sut.get_predictions()
        # cpp_preds returns {idx: (start_logits, end_logits)} tuples
        return {idx: pred for idx, pred in cpp_preds.items()}

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._cpp_sut.set_store_predictions(store)
        self._is_accuracy_mode = store

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy metrics (F1 and Exact Match).

        Uses SQuAD dataset's postprocess and compute_accuracy methods.
        """
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        # Sort by index
        sorted_indices = sorted(predictions.keys())

        # Prepare results for postprocess
        results = [(predictions[idx][0], predictions[idx][1]) for idx in sorted_indices]

        # Use dataset's postprocess to get answer texts
        pred_texts = self.qsl.dataset.postprocess(results, sorted_indices)

        # Compute F1 and EM
        return self.qsl.dataset.compute_accuracy(pred_texts, sorted_indices)

    def reset(self) -> None:
        """Reset state."""
        self._cpp_sut.reset_counters()
        self._query_count = 0
        self._server_callback_set = False
        self._qsl_validated = False
        try:
            self._cpp_sut.clear_sample_data()
        except Exception:
            pass
        try:
            self._cpp_sut.enable_direct_loadgen(False)
        except Exception:
            pass
        self._cpp_qsl_registered = False
        self._qsl_data_ready = False


def is_bert_multi_die_cpp_available() -> bool:
    """Check if C++ BERT multi-die SUT is available."""
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
