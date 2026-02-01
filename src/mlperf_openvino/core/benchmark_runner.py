"""MLPerf OpenVINO Benchmark runner."""

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
from .sut_factory import SUTFactory
from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.base import QuerySampleLibrary

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main class for running MLPerf benchmarks."""

    def __init__(self, config: BenchmarkConfig):
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
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)

        model_type = self.config.model.model_type
        is_sdxl = model_type == ModelType.SDXL
        is_whisper = model_type == ModelType.WHISPER
        is_accelerator = self.config.openvino.is_accelerator_device()
        uses_cpp_multi_die_sut = (
            model_type in (ModelType.RESNET50, ModelType.BERT, ModelType.RETINANET)
            and is_accelerator
        )

        if is_sdxl:
            self.backend = None
        elif is_whisper:
            model_path = Path(self.config.model.model_path) if self.config.model.model_path else None
            if model_path and model_path.is_dir():
                self.backend = None
            else:
                self.backend = self._create_backend()
        elif uses_cpp_multi_die_sut:
            self.backend = None
            logger.info("Skipping Python backend (C++ SUT will compile model)")
        else:
            self.backend = self._create_backend()

        if is_sdxl:
            self._setup_sdxl()
        elif is_whisper:
            self._setup_whisper()
        elif model_type == ModelType.RESNET50:
            self._setup_resnet50()
        elif model_type == ModelType.BERT:
            self._setup_bert()
        elif model_type == ModelType.RETINANET:
            self._setup_retinanet()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_backend(self) -> Union["OpenVINOBackend", "MultiDeviceBackend"]:
        """Create the appropriate backend based on device configuration."""
        from ..backends.multi_device_backend import MultiDeviceBackend

        device = self.config.openvino.device

        use_nhwc = True
        if hasattr(self.config.model, 'preprocessing') and self.config.model.preprocessing:
            use_nhwc = getattr(self.config.model.preprocessing, 'output_layout', 'NHWC') == 'NHWC'

        if self.config.openvino.is_accelerator_device():
            if self.config.openvino.is_specific_die():
                target_devices = [device]
            else:
                target_devices = None
            backend = MultiDeviceBackend(
                model_path=self.config.model.model_path,
                config=self.config.openvino,
                target_devices=target_devices,
            )
            backend.load()
            return backend

        backend = OpenVINOBackend(
            model_path=self.config.model.model_path,
            config=self.config.openvino,
            use_nhwc_input=use_nhwc,
        )
        backend.load()
        return backend

    def _create_sut_for_backend(
        self,
        qsl: QuerySampleLibrary,
    ) -> Any:
        """Create appropriate SUT based on backend type."""
        from ..backends.multi_device_backend import MultiDeviceBackend
        from .sut import OpenVINOSUT

        if self.backend is None and self.config.openvino.is_accelerator_device():
            return SUTFactory.create_multi_die_sut(
                ModelType.RESNET50, self.config, qsl, self.backend
            )
        elif isinstance(self.backend, MultiDeviceBackend):
            return SUTFactory.create_multi_die_sut(
                ModelType.RESNET50, self.config, qsl, self.backend
            )
        else:
            logger.info(f"Using OpenVINOSUT on {self.config.openvino.device}")
            return OpenVINOSUT(
                config=self.config,
                backend=self.backend,
                qsl=qsl,
                scenario=self.config.scenario,
            )

    def _setup_resnet50(self) -> None:
        """Set up ResNet50 benchmark."""
        from ..datasets.imagenet import ImageNetQSL
        from .cpp_sut_wrapper import create_sut

        self.qsl = ImageNetQSL(
            data_path=self.config.dataset.path,
            val_map_path=self.config.dataset.val_map,
            preprocessing=self.config.model.preprocessing,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=1024,
        )
        self.qsl.load()

        if self.config.openvino.is_accelerator_device():
            self.sut = self._create_sut_for_backend(self.qsl)
        else:
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

        self.qsl = SQuADQSL(
            data_path=self.config.dataset.path,
            vocab_file=self.config.dataset.val_map,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=10833,
        )
        self.qsl.load()

        if self.config.openvino.is_accelerator_device():
            self.sut = SUTFactory.create_multi_die_sut(
                ModelType.BERT, self.config, self.qsl, self.backend
            )
        else:
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

        output_layout = "NHWC"  # Default NHWC, model handles conversion via PrePostProcessor
        if hasattr(self.config.model, 'preprocessing') and self.config.model.preprocessing:
            output_layout = getattr(self.config.model.preprocessing, 'output_layout', 'NHWC')

        self.qsl = OpenImagesQSL(
            data_path=self.config.dataset.path,
            annotations_file=self.config.dataset.val_map,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            # performance_sample_count uses MLPerf official default (64) from OpenImagesQSL
            output_layout=output_layout,
        )
        self.qsl.load()

        if self.config.openvino.is_accelerator_device():
            self.sut = SUTFactory.create_multi_die_sut(
                ModelType.RETINANET, self.config, self.qsl, self.backend
            )
        else:
            self.sut = create_retinanet_sut(
                config=self.config,
                model_path=self.config.model.model_path,
                qsl=self.qsl,
                scenario=self.config.scenario,
            )

    def _setup_whisper(self) -> None:
        """Set up Whisper benchmark."""
        from ..datasets.librispeech import LibriSpeechQSL

        self.qsl = LibriSpeechQSL(
            data_path=self.config.dataset.path,
            transcript_path=self.config.dataset.val_map,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=2513,
        )
        self.qsl.load()

        model_path = Path(self.config.model.model_path)

        encoder_path = None
        decoder_path = None

        if model_path.is_dir():
            encoder_candidates = [
                model_path / "encoder_model.xml",
                model_path / "openvino_encoder_model.xml",
            ]
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

        if self.config.openvino.is_accelerator_device():
            if encoder_path and decoder_path:
                from .whisper_sut import WhisperMultiDieSUT
                logger.info(f"Using Whisper multi-die SUT on {self.config.openvino.device}")
                self.sut = WhisperMultiDieSUT(
                    config=self.config,
                    encoder_path=encoder_path,
                    decoder_path=decoder_path,
                    qsl=self.qsl,
                    scenario=self.config.scenario,
                )
                return
            else:
                raise ValueError(
                    f"Whisper multi-die SUT requires encoder and decoder models. "
                    f"Found encoder: {encoder_path}, decoder: {decoder_path}"
                )

        try:
            from .whisper_sut import WhisperOptimumSUT, OPTIMUM_AVAILABLE

            if OPTIMUM_AVAILABLE and model_path.is_dir():
                config_file = model_path / "config.json"
                if config_file.exists():
                    self.sut = WhisperOptimumSUT(
                        config=self.config,
                        model_path=model_path,
                        qsl=self.qsl,
                        scenario=self.config.scenario,
                    )
                    return
        except Exception:
            pass

        from .whisper_sut import WhisperSUT, WhisperEncoderOnlySUT

        if encoder_path and decoder_path:
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
            xml_files = list(model_path.glob("*.xml"))
            raise ValueError(
                f"Whisper model directory {model_path} missing encoder/decoder. "
                f"Found: {[f.name for f in xml_files]}"
            )

        if self.backend is None:
            raise ValueError(f"Cannot load Whisper model from {model_path}")

        self.sut = WhisperEncoderOnlySUT(
            config=self.config,
            backend=self.backend,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _setup_sdxl(self) -> None:
        """Set up Stable Diffusion XL benchmark."""
        from ..datasets.coco_prompts import COCOPromptsQSL

        self.qsl = COCOPromptsQSL(
            data_path=self.config.dataset.path,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=5000,
        )
        self.qsl.load()

        model_path = Path(self.config.model.model_path)

        try:
            from .sdxl_sut import SDXLOptimumSUT, OPTIMUM_SDXL_AVAILABLE

            if OPTIMUM_SDXL_AVAILABLE and model_path.is_dir():
                config_file = model_path / "model_index.json"
                if not config_file.exists():
                    config_file = model_path / "config.json"

                if config_file.exists():
                    self.sut = SDXLOptimumSUT(
                        config=self.config,
                        model_path=model_path,
                        qsl=self.qsl,
                        scenario=self.config.scenario,
                    )
                    return
        except Exception:
            pass

        from .sdxl_sut import SDXLManualSUT
        self.sut = SDXLManualSUT(
            config=self.config,
            model_path=model_path,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )

    def _get_test_settings(self) -> "lg.TestSettings":
        """Create LoadGen test settings."""
        settings = lg.TestSettings()

        if self.config.scenario == Scenario.OFFLINE:
            settings.scenario = lg.TestScenario.Offline
        elif self.config.scenario == Scenario.SERVER:
            settings.scenario = lg.TestScenario.Server
        else:
            raise ValueError(f"Unsupported scenario: {self.config.scenario}")

        if self.config.test_mode == TestMode.ACCURACY_ONLY:
            settings.mode = lg.TestMode.AccuracyOnly
        elif self.config.test_mode == TestMode.PERFORMANCE_ONLY:
            settings.mode = lg.TestMode.PerformanceOnly
        elif self.config.test_mode == TestMode.FIND_PEAK_PERFORMANCE:
            settings.mode = lg.TestMode.FindPeakPerformance
        else:
            settings.mode = lg.TestMode.PerformanceOnly

        scenario_config = self.config.get_scenario_config()
        settings.min_duration_ms = scenario_config.min_duration_ms
        settings.min_query_count = scenario_config.min_query_count
        settings.qsl_rng_seed = scenario_config.qsl_rng_seed
        settings.sample_index_rng_seed = scenario_config.sample_index_rng_seed
        settings.schedule_rng_seed = scenario_config.schedule_rng_seed

        logger.info(f"LoadGen settings: min_duration={scenario_config.min_duration_ms/1000:.0f}s, min_query_count={scenario_config.min_query_count}")

        if self.config.scenario == Scenario.OFFLINE:
            expected_qps = scenario_config.target_qps if scenario_config.target_qps > 0 else 1000.0
            settings.offline_expected_qps = expected_qps
            logger.info(f"Offline expected QPS: {expected_qps}")
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

        test_settings = self._get_test_settings()
        log_settings = self._get_log_settings()

        is_accuracy_mode = self.config.test_mode == TestMode.ACCURACY_ONLY
        if hasattr(self.sut, 'set_store_predictions'):
            self.sut.set_store_predictions(is_accuracy_mode)

        start_time = time.time()

        # Check if SUT supports native C++ benchmark (bypasses Python in hot path)
        use_native = (
            hasattr(self.sut, 'supports_native_benchmark') and
            self.sut.supports_native_benchmark()
        )

        if use_native:
            self.sut.run_native_benchmark(test_settings, log_settings)
            sut_handle = None
            qsl_handle = None
        else:
            sut_handle = self.sut.get_sut()
            qsl_handle = self.sut.get_qsl()
            lg.StartTestWithLogSettings(
                sut_handle,
                qsl_handle,
                test_settings,
                log_settings
            )

        end_time = time.time()
        duration = end_time - start_time

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

        if sut_handle is not None:
            lg.DestroySUT(sut_handle)
        if qsl_handle is not None:
            lg.DestroyQSL(qsl_handle)

        return self._results

    def _save_mlperf_accuracy_log(self) -> None:
        """Save accuracy results in MLPerf-compatible format.

        Writes mlperf_log_accuracy.json to the results directory.
        This file is required for official MLPerf submission.
        """
        if not self._accuracy_results:
            return

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
        elif model_type == ModelType.SDXL:
            primary_metric = "clip_score"
            metric_value = self._accuracy_results.get("clip_score", 0.0)
        else:
            primary_metric = "accuracy"
            metric_value = 0.0

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
        elif model_type == ModelType.SDXL:
            self._compute_sdxl_accuracy()

    def _compute_resnet50_accuracy(self) -> None:
        """Compute ResNet50 accuracy (Top-1)."""
        predictions = self.sut.get_predictions()
        total_samples = self.qsl.total_sample_count

        if not predictions:
            logger.warning("No predictions found for accuracy computation")
            self._accuracy_results = {"top1_accuracy": 0.0, "correct": 0, "total": 0}
            return

        if len(predictions) != total_samples:
            logger.warning(
                f"Prediction count mismatch: got {len(predictions)}, expected {total_samples}"
            )

        predicted_labels = []
        ground_truth = []

        for sample_idx, result in sorted(predictions.items()):
            pred_classes = self.qsl.dataset.postprocess(result, [sample_idx])
            predicted_labels.append(pred_classes[0])

            gt_label = self.qsl.get_label(sample_idx)
            ground_truth.append(gt_label)

        self._accuracy_results = self.qsl.dataset.compute_accuracy(
            predicted_labels,
            ground_truth
        )

        acc = self._accuracy_results.get('top1_accuracy', 0)
        correct = self._accuracy_results.get('correct', 0)
        total = self._accuracy_results.get('total', 0)
        logger.info(f"Top-1 Accuracy: {acc:.4f} ({correct}/{total})")

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

    def _compute_sdxl_accuracy(self) -> None:
        """Compute SDXL accuracy (CLIP Score and FID - MLPerf v5.1 metrics)."""
        self._accuracy_results = self.sut.compute_accuracy()

        clip_score = self._accuracy_results.get('clip_score', 0.0)
        fid_score = self._accuracy_results.get('fid_score', 0.0)

        logger.info(f"CLIP Score: {clip_score:.4f}")
        logger.info(f"FID Score: {fid_score:.4f}")

        # MLPerf v5.1 thresholds for SDXL
        # CLIP_SCORE: >= 31.68632 and <= 31.81332
        # FID_SCORE: >= 23.01086 and <= 23.95007
        clip_valid = 31.68632 <= clip_score <= 31.81332
        fid_valid = 23.01086 <= fid_score <= 23.95007

        if clip_valid and fid_valid:
            logger.info("SDXL accuracy: PASSED (within MLPerf thresholds)")
        else:
            logger.warning("SDXL accuracy: Outside MLPerf thresholds")
            if not clip_valid:
                logger.warning(f"  CLIP Score {clip_score:.4f} not in [31.68632, 31.81332]")
            if not fid_valid:
                logger.warning(f"  FID Score {fid_score:.4f} not in [23.01086, 23.95007]")

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
        """Print accuracy summary to console."""
        if "accuracy" not in self._results:
            return

        acc = self._results["accuracy"]
        model_type = self._results.get('model_type', '')
        scenario = self._results.get('scenario', 'N/A')

        print("\n" + "="*50)
        print(f"[Accuracy] {self._results.get('model', 'N/A')} / {scenario}")
        print("="*50)

        if model_type == 'resnet50':
            accuracy = acc.get('top1_accuracy', 0)
            correct = acc.get('correct', 0)
            total = acc.get('total', 0)
            # MLPerf ResNet50 threshold: 75.69% (99% of 76.46%)
            status = "PASS" if accuracy >= 0.7569 else "FAIL"
            print(f"Top-1: {accuracy:.4f} ({correct}/{total}) [{status}]")
        elif model_type == 'bert':
            f1 = acc.get('f1', 0)
            em = acc.get('exact_match', 0)
            # MLPerf BERT threshold: F1 >= 89.965 (99% of 90.874)
            status = "PASS" if f1 >= 89.965 else "FAIL"
            print(f"F1: {f1:.2f} [{status}]")
            print(f"EM: {em:.2f}")
        elif model_type == 'retinanet':
            mAP = acc.get('mAP', 0)
            # MLPerf RetinaNet threshold: 37.19% mAP (99% of 37.57%)
            status = "PASS" if mAP >= 0.3719 else "FAIL"
            print(f"mAP: {mAP:.4f} [{status}]")
        elif model_type == 'whisper':
            word_acc = acc.get('word_accuracy', 0)
            wer = acc.get('wer', 0)
            # MLPerf Whisper threshold: 96.94% word accuracy (99% of 97.93%)
            status = "PASS" if word_acc >= 0.9694 else "FAIL"
            print(f"Word Accuracy: {word_acc:.4f} [{status}]")
            print(f"WER: {wer:.4f}")
        elif model_type == 'sdxl':
            clip = acc.get('clip_score', 0)
            fid = acc.get('fid_score', 0)
            # MLPerf SDXL: CLIP >= 31.68632 and FID within range
            clip_pass = clip >= 31.68632
            status = "PASS" if clip_pass else "FAIL"
            print(f"CLIP: {clip:.4f} [{status}]")
            print(f"FID: {fid:.4f}")

        print("="*50 + "\n")
