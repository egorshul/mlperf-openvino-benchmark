"""
Main benchmark runner for MLPerf OpenVINO Benchmark.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .config import BenchmarkConfig, Scenario, TestMode
from .sut import OpenVINOSUT, AsyncOpenVINOSUT
from ..backends.openvino_backend import OpenVINOBackend
from ..datasets.imagenet import ImageNetQSL

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main class for running MLPerf benchmarks.
    
    This class orchestrates:
    - Model loading
    - Dataset preparation
    - Benchmark execution
    - Results collection and reporting
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
        self.qsl: Optional[ImageNetQSL] = None
        self.sut: Optional[OpenVINOSUT] = None
        
        self._results: Dict[str, Any] = {}
        self._accuracy_results: Dict[str, Any] = {}
    
    def setup(self) -> None:
        """Set up benchmark components."""
        logger.info("Setting up benchmark...")
        
        # Create output directories
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        logger.info(f"Loading model from {self.config.model.model_path}")
        self.backend = OpenVINOBackend(
            model_path=self.config.model.model_path,
            config=self.config.openvino,
        )
        self.backend.load()
        
        # Initialize dataset
        logger.info(f"Loading dataset from {self.config.dataset.path}")
        self.qsl = ImageNetQSL(
            data_path=self.config.dataset.path,
            val_map_path=self.config.dataset.val_map,
            preprocessing=self.config.model.preprocessing,
            count=self.config.dataset.num_samples if self.config.dataset.num_samples > 0 else None,
            performance_sample_count=1024,
        )
        self.qsl.load()
        
        # Initialize SUT
        logger.info(f"Creating SUT for scenario: {self.config.scenario}")
        self.sut = OpenVINOSUT(
            config=self.config,
            backend=self.backend,
            qsl=self.qsl,
            scenario=self.config.scenario,
        )
        
        logger.info("Setup complete")
    
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
        
        if self.config.scenario == Scenario.OFFLINE:
            if scenario_config.target_qps > 0:
                settings.offline_expected_qps = scenario_config.target_qps
        elif self.config.scenario == Scenario.SERVER:
            if scenario_config.target_latency_ns > 0:
                settings.server_target_latency_ns = scenario_config.target_latency_ns
            if scenario_config.target_qps > 0:
                settings.server_target_qps = scenario_config.target_qps
        
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
        
        lg.DestroySUT(sut_handle)
        lg.DestroyQSL(qsl_handle)
        
        return self._results
    
    def _compute_accuracy(self) -> None:
        """Compute accuracy metrics from predictions."""
        if self.sut is None or self.qsl is None:
            return
        
        predictions = self.sut.get_predictions()
        
        predicted_labels = []
        ground_truth = []
        
        for sample_idx, result in predictions.items():
            pred_class = int(np.argmax(result))
            predicted_labels.append(pred_class)
            
            gt_label = self.qsl.get_label(sample_idx)
            ground_truth.append(gt_label)
        
        self._accuracy_results = self.qsl.dataset.compute_accuracy(
            predicted_labels,
            ground_truth
        )
        
        logger.info(f"Accuracy: {self._accuracy_results['top1_accuracy']:.4f}")
    
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
            json.dump(self._results, f, indent=2)
        
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
        print(f"Scenario:  {self._results.get('scenario', 'N/A')}")
        print(f"Device:    {self._results.get('device', 'N/A')}")
        print("-"*60)
        print(f"Duration:  {self._results.get('duration_seconds', 0):.2f} seconds")
        print(f"Samples:   {self._results.get('samples_processed', 0)}")
        print(f"Throughput: {self._results.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        
        if "accuracy" in self._results:
            print("-"*60)
            acc = self._results["accuracy"]
            print(f"Top-1 Accuracy: {acc.get('top1_accuracy', 0):.4f} ({acc.get('correct', 0)}/{acc.get('total', 0)})")
        
        print("="*60 + "\n")
