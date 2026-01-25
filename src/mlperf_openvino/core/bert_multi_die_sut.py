"""
BERT SUT wrapper for multi-die NPU accelerators.

Sequence buckets: [128, 165, 256, 384]
Offline: batch [4,4,2,2] for throughput
Server: batch=1 for latency
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

try:
    from ..cpp import BertMultiDieSUT as CppBertMultiDieSUT, CPP_AVAILABLE
    CPP_BERT_AVAILABLE = CPP_AVAILABLE and CppBertMultiDieSUT is not None
except ImportError:
    CPP_BERT_AVAILABLE = False
    CppBertMultiDieSUT = None

from ..datasets.squad import SQuADQSL
from .config import BenchmarkConfig, Scenario

logger = logging.getLogger(__name__)

SEQ_BUCKETS = [128, 165, 256, 384]
DEFAULT_NIREQ = 8


class BertMultiDieSUTWrapper:
    """BERT SUT wrapper for multi-die NPU accelerators."""

    def __init__(
        self,
        config: BenchmarkConfig,
        qsl: SQuADQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen not installed")

        if not CPP_BERT_AVAILABLE:
            raise ImportError("C++ BERT SUT not available")

        self.config = config
        self.qsl = qsl
        self.scenario = scenario

        compile_props = {}
        if hasattr(config.openvino, 'device_properties') and config.openvino.device_properties:
            compile_props = dict(config.openvino.device_properties)

        nireq = DEFAULT_NIREQ
        if hasattr(config.model, 'nireq_per_config'):
            nireq = config.model.nireq_per_config
        elif hasattr(config.model, 'server') and config.model.server:
            nireq = getattr(config.model.server, 'nireq_per_config', DEFAULT_NIREQ)

        device_prefix = config.openvino.get_device_prefix()
        self._cpp_sut = CppBertMultiDieSUT(
            config.model.model_path,
            device_prefix,
            compile_props,
            nireq
        )

        if config.openvino.is_specific_die():
            self._cpp_sut.set_target_devices([config.openvino.device])

        if scenario == Scenario.SERVER:
            self._cpp_sut.set_server_mode(True)

        self._is_loaded = False
        self._is_accuracy_mode = False
        self._start_time = 0.0
        self._query_count = 0
        self._samples_registered = False

    def load(self, is_accuracy_mode: bool = False) -> None:
        self._cpp_sut.load()
        self._cpp_sut.warmup(2)
        self._cpp_sut.set_store_predictions(is_accuracy_mode)
        self._is_accuracy_mode = is_accuracy_mode
        self._is_loaded = True

        configs = self._cpp_sut.get_model_configs()
        logger.info(f"Loaded {len(configs)} models: {configs}")
        logger.info(f"Devices: {self._cpp_sut.get_active_devices()}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_dies(self) -> int:
        return self._cpp_sut.get_num_dies()

    def _register_samples(self) -> None:
        if self._samples_registered:
            return

        if not hasattr(self.qsl, '_loaded_samples'):
            return

        count = 0
        for idx, data in self.qsl._loaded_samples.items():
            input_ids = data.get('input_ids')
            attention_mask = data.get('attention_mask')
            token_type_ids = data.get('token_type_ids')

            if input_ids is None or attention_mask is None or token_type_ids is None:
                continue

            seq_len = self.qsl.get_actual_seq_len(idx)
            self._cpp_sut.register_sample(idx, input_ids, attention_mask, token_type_ids, seq_len)
            count += 1

        self._samples_registered = True
        logger.debug(f"Registered {count} samples")

    def _issue_query_offline(self, query_samples: List) -> None:
        self._start_time = time.time()
        self._cpp_sut.reset_counters()
        self._cpp_sut.enable_direct_loadgen(True)

        self._register_samples()

        buckets: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(SEQ_BUCKETS))}

        for qs in query_samples:
            bucket_idx = self.qsl.get_sample_bucket(qs.index)
            buckets[bucket_idx].append((qs.id, qs.index))

        for bucket_idx, samples in buckets.items():
            if not samples:
                continue
            query_ids = [s[0] for s in samples]
            sample_indices = [s[1] for s in samples]
            self._cpp_sut.submit_batch(bucket_idx, query_ids, sample_indices)

        self._cpp_sut.wait_all()
        self._query_count += 1

    def _issue_query_server(self, query_samples: List) -> None:
        if not self._samples_registered:
            self._register_samples()
            self._cpp_sut.stage_samples()
            self._cpp_sut.enable_direct_loadgen(True)

        query_ids = [qs.id for qs in query_samples]
        sample_indices = [qs.index for qs in query_samples]
        self._cpp_sut.issue_queries(query_ids, sample_indices)
        self._query_count += 1

    def issue_queries(self, query_samples: List) -> None:
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def flush_queries(self) -> None:
        self._cpp_sut.wait_all()

    def get_sut(self):
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def get_qsl(self):
        return lg.ConstructQSL(
            self.qsl.total_sample_count,
            self.qsl.performance_sample_count,
            self.qsl.load_query_samples,
            self.qsl.unload_query_samples,
        )

    @property
    def name(self) -> str:
        return f"BertMultiDie-{self.num_dies}dies"

    @property
    def _sample_count(self) -> int:
        return self._cpp_sut.get_completed_count()

    def get_predictions(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        return {idx: pred for idx, pred in self._cpp_sut.get_predictions().items()}

    def set_store_predictions(self, store: bool) -> None:
        self._cpp_sut.set_store_predictions(store)
        self._is_accuracy_mode = store

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            return {'f1': 0.0, 'exact_match': 0.0, 'num_samples': 0}

        sorted_indices = sorted(predictions.keys())
        results = [(predictions[idx][0], predictions[idx][1]) for idx in sorted_indices]

        pred_texts = self.qsl.dataset.postprocess(results, sorted_indices)
        return self.qsl.dataset.compute_accuracy(pred_texts, sorted_indices)

    def reset(self) -> None:
        self._cpp_sut.reset_counters()
        self._query_count = 0
        self._samples_registered = False
        self._cpp_sut.clear_samples()
        self._cpp_sut.clear_predictions()


def is_bert_multi_die_cpp_available() -> bool:
    return CPP_BERT_AVAILABLE and LOADGEN_AVAILABLE
