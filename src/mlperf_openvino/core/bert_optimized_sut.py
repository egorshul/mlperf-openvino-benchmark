"""
Python wrapper for optimized BERT multi-die SUT with dynamic sequence length buckets.

Key optimizations:
- Sequence length buckets: [128, 165, 256, 384]
- Optimal batch sizes per bucket
- Multiple pre-compiled models
- Smart batching by sequence length
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
    from ..cpp import BertOptimizedSUT as CppBertOptimizedSUT, CPP_AVAILABLE
    CPP_OPTIMIZED_AVAILABLE = CPP_AVAILABLE and CppBertOptimizedSUT is not None
except ImportError:
    CPP_OPTIMIZED_AVAILABLE = False
    CppBertOptimizedSUT = None

from ..datasets.squad import SQuADQSL
from .config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)

# Sequence length buckets - must match C++ constants
SEQ_BUCKETS = [128, 165, 256, 384]
# More aggressive default batch sizes for better throughput
DEFAULT_BATCH_SIZES = [8, 8, 4, 4]  # Was [4, 4, 2, 2]
DEFAULT_NIREQ_PER_CONFIG = 8  # Was 4


class BertOptimizedSUTWrapper:
    """
    Wrapper for optimized C++ BERT SUT with dynamic sequence length buckets.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: SQuADQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen not installed")

        if not CPP_OPTIMIZED_AVAILABLE:
            raise ImportError(
                "C++ optimized BERT SUT not available. Build with: ./build_cpp.sh"
            )

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        # Build compile properties
        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = dict(config.openvino.device_properties)

        # Get nireq from config (higher = more throughput for Offline)
        nireq_per_config = DEFAULT_NIREQ_PER_CONFIG
        if hasattr(config.model, 'nireq_per_config'):
            nireq_per_config = config.model.nireq_per_config
        elif hasattr(config.model, 'server') and config.model.server:
            nireq_per_config = getattr(config.model.server, 'nireq_per_config', DEFAULT_NIREQ_PER_CONFIG)

        # Create C++ SUT
        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = CppBertOptimizedSUT(
            config.model.model_path,
            device_prefix,
            compile_props,
            nireq_per_config
        )

        # Set target devices if specific die
        if config.openvino.is_specific_die():
            self._cpp_sut.set_target_devices([config.openvino.device])

        # Configure batch sizes per bucket
        batch_sizes = DEFAULT_BATCH_SIZES.copy()
        if hasattr(config.model, 'bert_batch_sizes'):
            batch_sizes = config.model.bert_batch_sizes
        self._cpp_sut.set_bucket_batch_sizes(batch_sizes)

        # State
        self._is_loaded = False
        self._is_accuracy_mode = False
        self._start_time = 0.0
        self._query_count = 0
        self._samples_registered = False

        logger.debug(f"BertOptimizedSUTWrapper: device_prefix={device_prefix}")

    def load(self, is_accuracy_mode: bool = False) -> None:
        """Load and compile all model variants."""
        self._cpp_sut.load()
        self._cpp_sut.set_store_predictions(is_accuracy_mode)
        self._is_accuracy_mode = is_accuracy_mode
        self._is_loaded = True

        configs = self._cpp_sut.get_model_configs()
        logger.info(f"Loaded {len(configs)} model configs: {configs}")
        logger.info(f"Active devices: {self._cpp_sut.get_active_devices()}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _register_samples(self) -> None:
        """Register all loaded samples with sequence length info."""
        if self._samples_registered:
            return

        if not hasattr(self.qsl, '_loaded_samples'):
            logger.warning("QSL has no _loaded_samples")
            return

        count = 0
        for idx, data in self.qsl._loaded_samples.items():
            input_ids = data.get('input_ids')
            attention_mask = data.get('attention_mask')
            token_type_ids = data.get('token_type_ids')

            if input_ids is None or attention_mask is None or token_type_ids is None:
                continue

            # Get actual sequence length
            actual_seq_len = self.qsl.get_actual_seq_len(idx)

            # Register with C++
            self._cpp_sut.register_sample(
                idx, input_ids, attention_mask, token_type_ids, actual_seq_len
            )
            count += 1

        self._samples_registered = True
        logger.debug(f"Registered {count} samples with sequence length info")

    def _issue_query_offline(self, query_samples: List) -> None:
        """Process queries in Offline mode with bucket-aware batching."""
        self._start_time = time.time()
        self._cpp_sut.reset_counters()
        self._cpp_sut.enable_direct_loadgen(True)

        # Ensure samples are registered
        self._register_samples()

        # Group samples by bucket
        buckets: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(SEQ_BUCKETS))}

        for qs in query_samples:
            bucket_idx = self.qsl.get_sample_bucket(qs.index)
            buckets[bucket_idx].append((qs.id, qs.index))

        # Log bucket distribution
        total_samples = sum(len(b) for b in buckets.values())
        print(f"\n[BERT Optimized] Bucket distribution ({total_samples} samples):")
        for bucket_idx, samples in buckets.items():
            pct = 100 * len(samples) / total_samples if total_samples > 0 else 0
            print(f"  Bucket {bucket_idx} (seq<={SEQ_BUCKETS[bucket_idx]}): {len(samples):5d} samples ({pct:5.1f}%)")

        # Submit each bucket
        for bucket_idx, samples in buckets.items():
            if not samples:
                continue

            query_ids = [s[0] for s in samples]
            sample_indices = [s[1] for s in samples]

            self._cpp_sut.submit_batch(bucket_idx, query_ids, sample_indices)

        self._cpp_sut.wait_all()
        self._query_count += 1

        elapsed = time.time() - self._start_time
        completed = self._cpp_sut.get_completed_count()
        throughput = completed / elapsed if elapsed > 0 else 0
        print(f"\n[BERT Optimized] Offline: {completed} samples in {elapsed:.2f}s")
        print(f"[BERT Optimized] Throughput: {throughput:.1f} samples/sec")
        print(f"[BERT Optimized] Dies: {self.num_dies}, Configs: {self._cpp_sut.get_model_configs()}")

    def _issue_query_server(self, query_samples: List) -> None:
        """Process queries in Server mode."""
        if not self._samples_registered:
            self._register_samples()

        query_ids = [qs.id for qs in query_samples]
        sample_indices = [qs.index for qs in query_samples]

        self._cpp_sut.issue_queries(query_ids, sample_indices)
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
        return f"BertOptimized-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get stored predictions."""
        cpp_preds = self._cpp_sut.get_predictions()
        return {idx: pred for idx, pred in cpp_preds.items()}

    def set_store_predictions(self, store: bool) -> None:
        """Enable/disable prediction storage."""
        self._cpp_sut.set_store_predictions(store)
        self._is_accuracy_mode = store

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute accuracy metrics (F1 and Exact Match)."""
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        sorted_indices = sorted(predictions.keys())
        results = [(predictions[idx][0], predictions[idx][1]) for idx in sorted_indices]

        pred_texts = self.qsl.dataset.postprocess(results, sorted_indices)
        return self.qsl.dataset.compute_accuracy(pred_texts, sorted_indices)

    def reset(self) -> None:
        """Reset state."""
        self._cpp_sut.reset_counters()
        self._query_count = 0
        self._samples_registered = False
        self._cpp_sut.clear_samples()
        self._cpp_sut.clear_predictions()


def is_bert_optimized_cpp_available() -> bool:
    """Check if optimized C++ BERT SUT is available."""
    return CPP_OPTIMIZED_AVAILABLE and LOADGEN_AVAILABLE
