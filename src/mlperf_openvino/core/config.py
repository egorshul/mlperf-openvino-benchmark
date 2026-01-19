"""
Configuration management for MLPerf OpenVINO Benchmark.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class Scenario(Enum):
    """MLPerf supported scenarios."""
    OFFLINE = "Offline"
    SERVER = "Server"
    SINGLE_STREAM = "SingleStream"
    MULTI_STREAM = "MultiStream"


class TestMode(Enum):
    """MLPerf test modes."""
    ACCURACY_ONLY = "AccuracyOnly"
    PERFORMANCE_ONLY = "PerformanceOnly"
    FIND_PEAK_PERFORMANCE = "FindPeakPerformance"
    SUBMISSION_RUN = "SubmissionRun"


class ModelType(Enum):
    """Supported model types."""
    RESNET50 = "resnet50"
    BERT = "bert"
    RETINANET = "retinanet"
    WHISPER = "whisper"
    SDXL = "sdxl"


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    resize: Tuple[int, int] = (256, 256)
    center_crop: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (123.68, 116.78, 103.94)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_order: str = "RGB"


@dataclass
class OpenVINOConfig:
    """OpenVINO runtime configuration."""
    device: str = "CPU"
    num_streams: str = "AUTO"
    num_threads: int = 0  # 0 = auto-detect
    batch_size: int = 1  # Inference batch size
    enable_profiling: bool = False
    cache_dir: str = "./cache"
    performance_hint: str = "THROUGHPUT"  # THROUGHPUT or LATENCY
    bind_thread: bool = True
    threads_per_stream: int = 0
    enable_hyper_threading: bool = True
    input_layout: str = "NHWC"  # Input data layout: "NHWC" adds transpose for image models
    
    def to_properties(self) -> Dict[str, Any]:
        """Convert to OpenVINO properties dictionary."""
        properties = {
            "NUM_STREAMS": self.num_streams if self.num_streams != "AUTO" else "AUTO",
            "INFERENCE_NUM_THREADS": self.num_threads if self.num_threads > 0 else None,
            "CACHE_DIR": self.cache_dir,
            "PERFORMANCE_HINT": self.performance_hint,
        }
        
        # CPU-specific properties
        if self.device.upper() == "CPU":
            properties["AFFINITY"] = "CORE" if self.bind_thread else "NONE"
            if self.threads_per_stream > 0:
                properties["INFERENCE_THREADS_PER_STREAM"] = self.threads_per_stream
        
        # Remove None values
        return {k: v for k, v in properties.items() if v is not None}


@dataclass
class ScenarioConfig:
    """Scenario-specific configuration."""
    min_duration_ms: int = 60000
    min_query_count: int = 24576
    samples_per_query: int = 1
    target_latency_ns: int = 0
    target_qps: float = 0.0
    # MLPerf seeds for reproducibility
    qsl_rng_seed: int = 13281865557512327830
    sample_index_rng_seed: int = 198141574272810017
    schedule_rng_seed: int = 7575108116881280410


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    task: str
    model_type: ModelType
    input_shape: List[int]
    input_name: str = "input"
    output_name: str = "output"
    data_format: str = "NHWC"  # Default for image models
    dtype: str = "FP32"
    
    # Accuracy targets
    accuracy_target: float = 0.0
    accuracy_threshold: float = 0.99
    
    # Preprocessing
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Scenario configs
    offline: ScenarioConfig = field(default_factory=ScenarioConfig)
    server: ScenarioConfig = field(default_factory=ScenarioConfig)
    
    # Model paths
    model_path: Optional[str] = None
    onnx_url: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    path: str
    val_map: Optional[str] = None
    calibration_path: Optional[str] = None
    num_samples: int = 0


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    
    # General settings
    mlperf_version: str = "5.1"
    division: str = "open"
    category: str = "datacenter"
    
    # Model configuration
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="ResNet50",
        task="image_classification",
        model_type=ModelType.RESNET50,
        input_shape=[1, 3, 224, 224]
    ))
    
    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="imagenet2012",
        path="./data/imagenet"
    ))
    
    # OpenVINO configuration
    openvino: OpenVINOConfig = field(default_factory=OpenVINOConfig)
    
    # Test settings
    scenario: Scenario = Scenario.OFFLINE
    test_mode: TestMode = TestMode.PERFORMANCE_ONLY
    
    # Output settings
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    @classmethod
    def from_yaml(cls, yaml_path: str, model_name: str = "resnet50") -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parse global settings
        global_settings = data.get("global", {})
        
        # Parse model config
        model_data = data.get("models", {}).get(model_name, {})
        
        preprocessing_data = model_data.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            resize=tuple(preprocessing_data.get("resize", [256, 256])),
            center_crop=tuple(preprocessing_data.get("center_crop", [224, 224])),
            mean=tuple(preprocessing_data.get("mean", [123.68, 116.78, 103.94])),
            std=tuple(preprocessing_data.get("std", [1.0, 1.0, 1.0])),
            channel_order=preprocessing_data.get("channel_order", "RGB"),
        )
        
        offline_data = model_data.get("offline", {})
        offline = ScenarioConfig(
            min_duration_ms=offline_data.get("min_duration_ms", 60000),
            min_query_count=offline_data.get("min_query_count", 24576),
            samples_per_query=offline_data.get("samples_per_query", 1),
        )
        
        server_data = model_data.get("server", {})
        server = ScenarioConfig(
            min_duration_ms=server_data.get("min_duration_ms", 60000),
            min_query_count=server_data.get("min_query_count", 24576),
            target_latency_ns=server_data.get("target_latency_ns", 15000000),
            target_qps=server_data.get("target_qps", 10000.0),  # High default for max throughput
            # MLPerf seeds for reproducibility
            qsl_rng_seed=server_data.get("qsl_rng_seed", 13281865557512327830),
            sample_index_rng_seed=server_data.get("sample_index_rng_seed", 198141574272810017),
            schedule_rng_seed=server_data.get("schedule_rng_seed", 7575108116881280410),
        )
        
        sources = model_data.get("sources", {})
        
        model_config = ModelConfig(
            name=model_data.get("name", model_name),
            task=model_data.get("task", "image_classification"),
            model_type=ModelType(model_name),
            input_shape=model_data.get("input_shape", [1, 3, 224, 224]),
            input_name=model_data.get("input_name", "input"),
            output_name=model_data.get("output_name", "output"),
            data_format=model_data.get("data_format", "NHWC"),
            dtype=model_data.get("dtype", "FP32"),
            accuracy_target=model_data.get("accuracy_target", 0.0),
            accuracy_threshold=model_data.get("accuracy_threshold", 0.99),
            preprocessing=preprocessing,
            offline=offline,
            server=server,
            onnx_url=sources.get("onnx_url"),
        )
        
        # Parse OpenVINO config
        ov_data = data.get("openvino", {})
        cpu_data = ov_data.get("cpu", {})
        
        openvino_config = OpenVINOConfig(
            device=ov_data.get("device", "CPU"),
            num_streams=str(ov_data.get("num_streams", "AUTO")),
            num_threads=ov_data.get("num_threads", 0),
            batch_size=ov_data.get("batch_size", 1),
            enable_profiling=ov_data.get("enable_profiling", False),
            cache_dir=ov_data.get("cache_dir", "./cache"),
            performance_hint=ov_data.get("performance_hint", "THROUGHPUT"),
            bind_thread=cpu_data.get("bind_thread", True),
            threads_per_stream=cpu_data.get("threads_per_stream", 0),
            enable_hyper_threading=cpu_data.get("enable_hyper_threading", True),
        )
        
        # Parse dataset config
        datasets_data = data.get("datasets", {})
        dataset_name = model_data.get("dataset", "imagenet2012")
        dataset_data = datasets_data.get(dataset_name, {})
        
        dataset_config = DatasetConfig(
            name=dataset_name,
            path=dataset_data.get("path", f"./data/{dataset_name}"),
            val_map=dataset_data.get("val_map"),
            calibration_path=dataset_data.get("calibration_path"),
        )
        
        # Parse output config
        output_data = data.get("output", {})
        
        return cls(
            mlperf_version=global_settings.get("mlperf_version", "5.1"),
            division=global_settings.get("division", "open"),
            category=global_settings.get("category", "datacenter"),
            model=model_config,
            dataset=dataset_config,
            openvino=openvino_config,
            results_dir=output_data.get("results_dir", "./results"),
            logs_dir=output_data.get("logs_dir", "./logs"),
        )
    
    @classmethod
    def default_resnet50(cls) -> "BenchmarkConfig":
        """Create default ResNet50 configuration."""
        return cls(
            model=ModelConfig(
                name="ResNet50-v1.5",
                task="image_classification",
                model_type=ModelType.RESNET50,
                input_shape=[1, 224, 224, 3],  # NHWC format
                input_name="input",
                output_name="output",
                data_format="NHWC",
                dtype="FP32",
                accuracy_target=0.7646,
                accuracy_threshold=0.99,
                onnx_url="https://zenodo.org/record/4735647/files/resnet50_v1.onnx",
                offline=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=24576,  # MLPerf official
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=24576,  # Server uses min_duration primarily
                    target_latency_ns=15000000,  # 15ms
                    target_qps=10000.0,
                ),
            ),
            dataset=DatasetConfig(
                name="imagenet2012",
                path="./data/imagenet",
            ),
        )

    @classmethod
    def default_whisper(cls) -> "BenchmarkConfig":
        """Create default Whisper configuration."""
        return cls(
            model=ModelConfig(
                name="Whisper-Large-v3",
                task="speech_recognition",
                model_type=ModelType.WHISPER,
                input_shape=[1, 80, 3000],  # (batch, n_mels, time_frames)
                input_name="input_features",
                output_name="sequences",
                data_format="NCT",  # batch, channels (mels), time
                dtype="FP32",
                accuracy_target=0.979329,  # Word Accuracy (official MLPerf v5.1)
                accuracy_threshold=0.99,
                preprocessing=PreprocessingConfig(),  # Not used for audio
                offline=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=2513,  # MLPerf official (LibriSpeech)
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=2513,  # Server uses min_duration primarily
                    target_latency_ns=1000000000,  # 1 second for ASR
                    target_qps=500.0,
                ),
                onnx_url="https://huggingface.co/openai/whisper-large-v3/resolve/main/model.onnx",
            ),
            dataset=DatasetConfig(
                name="librispeech",
                path="./data/librispeech",
            ),
        )

    @classmethod
    def default_bert(cls) -> "BenchmarkConfig":
        """Create default BERT-Large configuration for SQuAD."""
        return cls(
            model=ModelConfig(
                name="BERT-Large",
                task="question_answering",
                model_type=ModelType.BERT,
                input_shape=[1, 384],  # (batch, seq_length)
                input_name="input_ids",
                output_name="output",
                data_format="NC",
                dtype="FP32",
                accuracy_target=0.90874,  # F1 score (official MLPerf v5.1)
                accuracy_threshold=0.99,
                preprocessing=PreprocessingConfig(),  # Not used for text
                offline=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=10833,  # MLPerf official (SQuAD dataset size)
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=10833,  # Server uses min_duration primarily
                    target_latency_ns=130000000,  # 130ms
                    target_qps=5000.0,
                ),
                onnx_url="https://zenodo.org/record/3733910/files/model.onnx",
            ),
            dataset=DatasetConfig(
                name="squad",
                path="./data/squad",
            ),
        )

    @classmethod
    def default_retinanet(cls) -> "BenchmarkConfig":
        """Create default RetinaNet configuration for OpenImages."""
        return cls(
            model=ModelConfig(
                name="RetinaNet",
                task="object_detection",
                model_type=ModelType.RETINANET,
                input_shape=[1, 800, 800, 3],  # NHWC format
                input_name="input",
                output_name="output",
                data_format="NHWC",
                dtype="FP32",
                accuracy_target=0.3757,  # mAP (official MLPerf v5.1)
                accuracy_threshold=0.99,
                preprocessing=PreprocessingConfig(
                    resize=(800, 800),
                    center_crop=(800, 800),
                    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                    std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
                    channel_order="RGB",
                ),
                offline=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=24576,  # MLPerf official
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=24576,  # Server uses min_duration primarily
                    target_latency_ns=100000000,  # 100ms
                    target_qps=1000.0,
                ),
                onnx_url="https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx",
            ),
            dataset=DatasetConfig(
                name="openimages",
                path="./data/openimages",
            ),
        )

    @classmethod
    def default_sdxl(cls) -> "BenchmarkConfig":
        """Create default Stable Diffusion XL configuration for COCO 2014."""
        return cls(
            model=ModelConfig(
                name="SDXL",
                task="text_to_image",
                model_type=ModelType.SDXL,
                input_shape=[1, 77],  # Tokenized prompt shape
                input_name="prompt",
                output_name="images",
                data_format="NCHW",  # Output images in NCHW
                dtype="FP16",  # SDXL typically uses FP16
                accuracy_target=23.5,  # FID score target (lower is better)
                accuracy_threshold=1.0,  # Allow ±2% variation
                offline=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=5000,  # MLPerf official: 5000 samples
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=60000,
                    min_query_count=5000,
                    target_latency_ns=20000000000,  # 20s per image
                    target_qps=1.0,
                ),
            ),
            dataset=DatasetConfig(
                name="coco2014",
                path="./data/coco2014",
            ),
        )

    def get_scenario_config(self) -> ScenarioConfig:
        """Get configuration for the current scenario."""
        if self.scenario == Scenario.OFFLINE:
            return self.model.offline
        elif self.scenario == Scenario.SERVER:
            return self.model.server
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check model path
        if self.model.model_path and not Path(self.model.model_path).exists():
            errors.append(f"Model file not found: {self.model.model_path}")
        
        # Check dataset path
        if not Path(self.dataset.path).exists():
            errors.append(f"Dataset path not found: {self.dataset.path}")
        
        # Check scenario support
        if self.scenario not in [Scenario.OFFLINE, Scenario.SERVER]:
            errors.append(f"Scenario {self.scenario} is not supported in this version")
        
        return errors
