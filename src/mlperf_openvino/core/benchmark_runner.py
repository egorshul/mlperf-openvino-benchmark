"""
Main benchmark runner for MLPerf OpenVINO Benchmark.

Supports multiple models: ResNet50, BERT, RetinaNet, Whisper.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, ModelType, Scenario, TestMode
from .sut import OpenVINOSUT
from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.base import QuerySampleLibrary

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main class for running MLPerf benchmarks.

    This class orchestrates:
    - Model loading
    - Dataset preparation
    - Benchmark execution
    - Results collection and reporting

    Supports models:
    - ResNet50 (Image Classification)
    - BERT-Large (Question Answering)
    - RetinaNet (Object Detection)
    - Whisper (Speech Recognition)
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError(
                "MLPerf LoadGen is not installed. Please install with: "
                "pip install mlcommons-loadgen"
            )

        self.config = config
        self.backend: Optional[OpenVINOBackend] = None
        self.qsl: Optional[QuerySampleLibrary] = None
        self.sut: Optional[Any] = None

        self._results: Dict[str, Any] = {}
        self._accuracy_results: Dict[str, Any] = {}

    def setup(self) -> None:
        """Set up benchmark components based on model type."""
        logger.info("Setting up benchmark...")

        # Create output directories
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model-specific components
        model_type = self.config.model.model_type

        # Initialize backend (skip for Whisper with directory path - it handles its own loading)
        model_path = Path(self.config.model.model_path)
        if model_type == ModelType.WHISPER and model_path.is_dir():
            logger.info(f"Whisper model directory: {self.config.model.model_path}")
            self.backend = None  # Will be set up in _setup_whisper
        else:
            logger.info(f"Loading model from {self.config.model.model_path}")
            self.backend = OpenVINOBackend(
                model_path=self.config.model.model_path,
                config=self.config.openvino,
            )
            self.backend.load()

        if model_type == ModelType.RESNET50:
            self._setup_resnet50()
        elif model_type == ModelType.BERT:
            self._setup_bert()
        elif model_type == ModelType.RETINANET:
            self._setup_retinanet()
        elif model_type == ModelType.WHISPER:
            self._setup_whisper()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info("Setup complete")

    def _setup_resnet50(self) -> None:
        """Set up ResNet50 benchmark."""
        from ..datasets.imagenet import ImageNetQSL
        from .cpp_sut_wrapper import create_sut

        logger.info(f"Loading ImageNet dataset from {self.config.dataset.path}")
        self.qsl = ImageNetQSL(
            data_path=self.config.dataset.path,
            val_map_path=self.config.dataset.val_map,
            preprocessing=self.config.model.preprocessing,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=1024,
            data_format=self.config.model.data_format,
        )
        self.qsl.load()

        logger.info(f"Creating SUT for scenario: {self.config.scenario}")
        # Use create_sut to automatically select C++ or Python SUT
        self.sut = create_sut(
            config=self.config,
            model_path=self.config.model.model_path,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _setup_bert(self) -> None:
        """Set up BERT benchmark."""
        from ..datasets.squad import SQuADQSL
        from .cpp_sut_wrapper import create_bert_sut

        logger.info(f"Loading SQuAD dataset from {self.config.dataset.path}")
        self.qsl = SQuADQSL(
            data_path=self.config.dataset.path,
            vocab_file=self.config.dataset.val_map,  # vocab file
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=10833,  # MLPerf default
        )
        self.qsl.load()

        logger.info(f"Creating BERT SUT for scenario: {self.config.scenario}")
        # Use create_bert_sut to automatically select C++ or Python SUT
        self.sut = create_bert_sut(
            config=self.config,
            model_path=self.config.model.model_path,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _setup_retinanet(self) -> None:
        """Set up RetinaNet benchmark."""
        from ..datasets.openimages import OpenImagesQSL
        from .cpp_sut_wrapper import create_retinanet_sut

        logger.info(f"Loading OpenImages dataset from {self.config.dataset.path}")
        self.qsl = OpenImagesQSL(
            data_path=self.config.dataset.path,
            annotations_file=self.config.dataset.val_map,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=24576,  # MLPerf default
            data_format=self.config.model.data_format,
        )
        self.qsl.load()

        logger.info(f"Creating RetinaNet SUT for scenario: {self.config.scenario}")
        # Use create_retinanet_sut to automatically select C++ or Python SUT
        self.sut = create_retinanet_sut(
            config=self.config,
            model_path=self.config.model.model_path,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _setup_whisper(self) -> None:
        """Set up Whisper benchmark."""
        from ..datasets.librispeech import LibriSpeechQSL

        logger.info(f"Loading LibriSpeech dataset from {self.config.dataset.path}")
        self.qsl = LibriSpeechQSL(
            data_path=self.config.dataset.path,
            transcript_path=self.config.dataset.val_map,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=2513,  # MLPerf default
        )
        self.qsl.load()

        model_path = Path(self.config.model.model_path)

        # Check for separate encoder/decoder models (try different naming conventions)
        encoder_path = None
        decoder_path = None

        if model_path.is_dir():
            # Try different naming conventions from optimum-cli
            encoder_candidates = [
                model_path / "encoder_model.xml",
                model_path / "openvino_encoder_model.xml",
            ]
            # Decoder candidates (Optimum handles KV-cache automatically)
            decoder_candidates = [
                model_path / "decoder_with_past_model.xml",
                model_path / "openvino_decoder_with_past_model.xml",
                model_path / "decoder_model_merged.xml",
                model_path / "openvino_decoder_model_merged.xml",
                model_path / "decoder_model.xml",
                model_path / "openvino_decoder_model.xml",
            ]

            for ep in encoder_candidates:
                if ep.exists():
                    encoder_path = ep
                    break

            for dp in decoder_candidates:
                if dp.exists():
                    decoder_path = dp
                    break

        # Use Optimum-Intel WhisperOptimumSUT (Python) - C++ SUT disabled due to KV-cache issues
        try:
            from .whisper_sut import WhisperOptimumSUT, OPTIMUM_AVAILABLE

            if OPTIMUM_AVAILABLE and model_path.is_dir():
                # Check if this looks like an optimum-exported model
                config_file = model_path / "config.json"
                if config_file.exists():
                    logger.info("Using Optimum-Intel for Whisper inference")
                    self.sut = WhisperOptimumSUT(
                        config=self.config,
                        model_path=model_path,
                        qsl=self.qsl,
                        scenario=self.config.scenario,
                    )
                    return
        except Exception as e:
            logger.warning(f"Could not use Optimum-Intel: {e}")
            logger.info("Falling back to manual encoder-decoder inference")

        # Fallback: Manual encoder-decoder inference
        from .whisper_sut import WhisperSUT, WhisperEncoderOnlySUT

        if encoder_path and decoder_path:
            logger.info(f"Setting up Whisper with separate encoder/decoder (Python)")
            logger.info(f"  Encoder: {encoder_path}")
            logger.info(f"  Decoder: {decoder_path}")

            encoder_backend = OpenVINOBackend(
                model_path=str(encoder_path),
                config=self.config.openvino,
            )
            encoder_backend.load()

            decoder_backend = OpenVINOBackend(
                model_path=str(decoder_path),
                config=self.config.openvino,
            )
            decoder_backend.load()

            self.sut = WhisperSUT(
                config=self.config,
                encoder_backend=encoder_backend,
                decoder_backend=decoder_backend,
                qsl=self.qsl,
                scenario=self.config.scenario,
            )
            return

        if model_path.is_dir():
            # List available files in directory
            xml_files = list(model_path.glob("*.xml"))
            logger.error(f"Could not find encoder/decoder models in {model_path}")
            logger.error(f"Available .xml files: {[f.name for f in xml_files]}")
            raise ValueError(
                f"Whisper model directory {model_path} does not contain "
                f"expected encoder/decoder files. Found: {[f.name for f in xml_files]}"
            )

        # Single model file - use encoder-only SUT
        if self.backend is None:
            raise ValueError(f"Cannot load Whisper model from {model_path}")

        logger.info(f"Creating Whisper encoder-only SUT for scenario: {self.config.scenario}")
        self.sut = WhisperEncoderOnlySUT(
            config=self.config,
            backend=self.backend,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _get_test_settings(self) -> "lg.TestSettings":
        """Create LoadGen test settings."""
        settings = lg.TestSettings()

        # Set scenario
        if self.config.scenario == Scenario.OFFLINE:
            settings.scenario = lg.TestScenario.Offline
        elif self.config.scenario == Scenario.SERVER:
            settings.scenario = lg.TestScenario.Server
        else:
            raise ValueError(f"Unsupported scenario: {self.config.scenario}")

        # Set mode
        if self.config.test_mode == TestMode.ACCURACY_ONLY:
            settings.mode = lg.TestMode.AccuracyOnly
        elif self.config.test_mode == TestMode.PERFORMANCE_ONLY:
            settings.mode = lg.TestMode.PerformanceOnly
        elif self.config.test_mode == TestMode.FIND_PEAK_PERFORMANCE:
            settings.mode = lg.TestMode.FindPeakPerformance
        else:
            settings.mode = lg.TestMode.PerformanceOnly

        # Get scenario-specific config
        scenario_config = self.config.get_scenario_config()

        # Set timing constraints
        settings.min_duration_ms = scenario_config.min_duration_ms
        settings.min_query_count = scenario_config.min_query_count

        # Set MLPerf seeds for reproducibility
        settings.qsl_rng_seed = scenario_config.qsl_rng_seed
        settings.sample_index_rng_seed = scenario_config.sample_index_rng_seed
        settings.schedule_rng_seed = scenario_config.schedule_rng_seed

        if self.config.scenario == Scenario.OFFLINE:
            # LoadGen requires expected_qps for Offline scenario
            # Use configured value or a reasonable default
            expected_qps = scenario_config.target_qps if scenario_config.target_qps > 0 else 1000.0
            settings.offline_expected_qps = expected_qps
        elif self.config.scenario == Scenario.SERVER:
            if scenario_config.target_latency_ns > 0:
                settings.server_target_latency_ns = scenario_config.target_latency_ns
            if scenario_config.target_qps > 0:
                settings.server_target_qps = scenario_config.target_qps
                logger.info(f"Server target QPS: {scenario_config.target_qps}")
            logger.info(f"Server target latency: {scenario_config.target_latency_ns / 1e6:.1f}ms")

        return settings

    def _get_log_settings(self) -> "lg.LogSettings":
        """Create LoadGen log settings."""
        log_settings = lg.LogSettings()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.logs_dir) / f"run_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_settings.log_output.outdir = str(log_dir)
        log_settings.log_output.copy_summary_to_stdout = True
        log_settings.enable_trace = False

        return log_settings

    def run(self) -> Dict[str, Any]:
        """Run the benchmark."""
        if self.sut is None:
            self.setup()

        logger.info(f"Running benchmark: {self.config.model.name}")
        logger.info(f"Scenario: {self.config.scenario}")
        logger.info(f"Mode: {self.config.test_mode}")

        test_settings = self._get_test_settings()
        log_settings = self._get_log_settings()

        # Log LoadGen settings for debugging
        scenario_config = self.config.get_scenario_config()
        logger.info(f"LoadGen settings:")
        logger.info(f"  min_duration_ms: {scenario_config.min_duration_ms}")
        logger.info(f"  min_query_count: {scenario_config.min_query_count}")
        if self.config.scenario == Scenario.SERVER:
            logger.info(f"  server_target_qps: {scenario_config.target_qps}")
            logger.info(f"  server_target_latency_ns: {scenario_config.target_latency_ns}")
        logger.info(f"  qsl_rng_seed: {scenario_config.qsl_rng_seed}")

        sut_handle = self.sut.get_sut()
        qsl_handle = self.sut.get_qsl()

        start_time = time.time()

        logger.info("Starting LoadGen test...")
        lg.StartTestWithLogSettings(
            sut_handle,
            qsl_handle,
            test_settings,
            log_settings
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Test completed in {duration:.2f} seconds")

        self._results = {
            "model": self.config.model.name,
            "model_type": self.config.model.model_type.value,
            "scenario": self.config.scenario.value,
            "mode": self.config.test_mode.value,
            "duration_seconds": duration,
            "samples_processed": self.sut._sample_count,
            "queries_processed": self.sut._query_count,
            "throughput_samples_per_sec": self.sut._sample_count / duration if duration > 0 else 0,
            "device": self.config.openvino.device,
            "timestamp": datetime.now().isoformat(),
        }

        if self.config.test_mode == TestMode.ACCURACY_ONLY:
            self._compute_accuracy()
            self._results["accuracy"] = self._accuracy_results
            self._save_mlperf_accuracy_log()

        lg.DestroySUT(sut_handle)
        lg.DestroyQSL(qsl_handle)

        return self._results

    def _save_mlperf_accuracy_log(self) -> None:
        """Save accuracy results in MLPerf-compatible format.

        Writes mlperf_log_accuracy.json to the results directory.
        This file is required for official MLPerf submission.
        """
        if not self._accuracy_results:
            return

        # Determine primary accuracy metric based on model type
        model_type = self.config.model.model_type
        if model_type == ModelType.RESNET50:
            primary_metric = "top1_accuracy"
            metric_value = self._accuracy_results.get("top1_accuracy", 0.0)
        elif model_type == ModelType.BERT:
            primary_metric = "f1"
            metric_value = self._accuracy_results.get("f1", 0.0)
        elif model_type == ModelType.RETINANET:
            primary_metric = "mAP"
            metric_value = self._accuracy_results.get("mAP", 0.0)
        elif model_type == ModelType.WHISPER:
            primary_metric = "word_accuracy"
            metric_value = self._accuracy_results.get("word_accuracy", 0.0)
        else:
            primary_metric = "accuracy"
            metric_value = 0.0

        # MLPerf accuracy log format
        accuracy_log = {
            "accuracy_results": {
                primary_metric: metric_value,
                **self._accuracy_results,
            },
            "model": self.config.model.name,
            "model_type": model_type.value if model_type else "unknown",
            "scenario": self.config.scenario.value,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to mlperf_log_accuracy.json in results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        accuracy_log_path = results_dir / "mlperf_log_accuracy.json"

        with open(accuracy_log_path, "w") as f:
            json.dump(accuracy_log, f, indent=2, default=str)

        logger.info(f"MLPerf accuracy log saved to {accuracy_log_path}")

    def _compute_accuracy(self) -> None:
        """Compute accuracy metrics based on model type."""
        if self.sut is None or self.qsl is None:
            return

        model_type = self.config.model.model_type

        if model_type == ModelType.RESNET50:
            self._compute_resnet50_accuracy()
        elif model_type == ModelType.BERT:
            self._compute_bert_accuracy()
        elif model_type == ModelType.RETINANET:
            self._compute_retinanet_accuracy()
        elif model_type == ModelType.WHISPER:
            self._compute_whisper_accuracy()

    def _compute_resnet50_accuracy(self) -> None:
        """Compute ResNet50 accuracy (Top-1)."""
        predictions = self.sut.get_predictions()

        predicted_labels = []
        ground_truth = []

        for sample_idx, result in sorted(predictions.items()):
            # Use dataset's postprocess to correctly handle model output format
            pred_classes = self.qsl.dataset.postprocess(result, [sample_idx])
            predicted_labels.append(pred_classes[0])

            gt_label = self.qsl.get_label(sample_idx)
            ground_truth.append(gt_label)

        self._accuracy_results = self.qsl.dataset.compute_accuracy(
            predicted_labels,
            ground_truth
        )

        logger.info(f"Top-1 Accuracy: {self._accuracy_results.get('top1_accuracy', 0):.4f}")

    def _compute_bert_accuracy(self) -> None:
        """Compute BERT accuracy (F1 and Exact Match)."""
        self._accuracy_results = self.sut.compute_accuracy()

        logger.info(f"F1 Score: {self._accuracy_results.get('f1', 0):.2f}")
        logger.info(f"Exact Match: {self._accuracy_results.get('exact_match', 0):.2f}")

    def _compute_retinanet_accuracy(self) -> None:
        """Compute RetinaNet accuracy (mAP)."""
        self._accuracy_results = self.sut.compute_accuracy()

        logger.info(f"mAP: {self._accuracy_results.get('mAP', 0):.4f}")

    def _compute_whisper_accuracy(self) -> None:
        """Compute Whisper accuracy (Word Accuracy - MLPerf v5.1 metric)."""
        predictions = self.sut.get_predictions()

        if not predictions:
            logger.error("No predictions found!")
            self._accuracy_results = {'word_accuracy': 0.0, 'wer': 0.0, 'num_samples': 0}
            return

        # Get predicted texts and ground truth
        pred_texts = []
        ground_truth = []

        for sample_idx in sorted(predictions.keys()):
            pred = predictions[sample_idx]
            if isinstance(pred, str):
                pred_texts.append(pred)
            else:
                pred_texts.append("")

            ground_truth.append(self.qsl.get_label(sample_idx))

        self._accuracy_results = self.qsl.dataset.compute_accuracy(pred_texts, ground_truth)

        logger.info(f"Word Accuracy: {self._accuracy_results.get('word_accuracy', 0):.4f}")
        logger.info(f"WER: {self._accuracy_results.get('wer', 0):.4f}")

    def run_accuracy(self) -> Dict[str, float]:
        """Run accuracy-only test."""
        original_mode = self.config.test_mode
        self.config.test_mode = TestMode.ACCURACY_ONLY

        try:
            self.run()
        finally:
            self.config.test_mode = original_mode

        return self._accuracy_results

    def run_performance(self) -> Dict[str, Any]:
        """Run performance-only test."""
        original_mode = self.config.test_mode
        self.config.test_mode = TestMode.PERFORMANCE_ONLY

        try:
            results = self.run()
        finally:
            self.config.test_mode = original_mode

        return results

    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save benchmark results to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.results_dir) / f"results_{timestamp}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        return str(output_path)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform

        try:
            import psutil
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "memory_gb": psutil.virtual_memory().total / (1024**3),
            }
        except ImportError:
            cpu_info = {}

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            **cpu_info,
        }

        if self.backend and self.backend.is_loaded:
            info["backend"] = self.backend.get_info()

        return info

    def print_summary(self) -> None:
        """Print benchmark summary to console."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Model:     {self._results.get('model', 'N/A')}")
        print(f"Type:      {self._results.get('model_type', 'N/A')}")
        print(f"Scenario:  {self._results.get('scenario', 'N/A')}")
        print(f"Device:    {self._results.get('device', 'N/A')}")
        print("-"*60)
        print(f"Duration:  {self._results.get('duration_seconds', 0):.2f} seconds")
        print(f"Samples:   {self._results.get('samples_processed', 0)}")
        print(f"Throughput: {self._results.get('throughput_samples_per_sec', 0):.2f} samples/sec")

        if "accuracy" in self._results:
            print("-"*60)
            acc = self._results["accuracy"]
            model_type = self._results.get('model_type', '')

            if model_type == 'resnet50':
                print(f"Top-1 Accuracy: {acc.get('top1_accuracy', 0):.4f} "
                      f"({acc.get('correct', 0)}/{acc.get('total', 0)})")
            elif model_type == 'bert':
                print(f"F1 Score: {acc.get('f1', 0):.2f}")
                print(f"Exact Match: {acc.get('exact_match', 0):.2f}")
            elif model_type == 'retinanet':
                print(f"mAP@0.5: {acc.get('mAP', 0):.4f}")
            elif model_type == 'whisper':
                print(f"Word Accuracy: {acc.get('word_accuracy', 0):.4f}")
                print(f"WER: {acc.get('wer', 0):.4f}")

        print("="*60 + "\n")
