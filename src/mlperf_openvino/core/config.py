"""Configuration for MLPerf OpenVINO Benchmark."""

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


SUPPORTED_SCENARIOS: Dict[ModelType, List["Scenario"]] = {
    ModelType.RESNET50: [Scenario.OFFLINE, Scenario.SERVER],
    ModelType.BERT: [Scenario.OFFLINE, Scenario.SERVER],
    ModelType.RETINANET: [Scenario.OFFLINE, Scenario.SERVER],
    ModelType.WHISPER: [Scenario.OFFLINE],
    ModelType.SDXL: [Scenario.OFFLINE, Scenario.SERVER],
}


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    resize: Tuple[int, int] = (256, 256)
    center_crop: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (123.68, 116.78, 103.94)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_order: str = "RGB"
    output_layout: str = "NHWC"  # "NCHW" or "NHWC" - NHWC is default, model handles conversion


@dataclass
class OpenVINOConfig:
    """OpenVINO runtime configuration."""
    device: str = "CPU"
    num_streams: str = "AUTO"
    num_threads: int = 0  # 0 = auto-detect
    batch_size: int = 0  # Inference batch size (0 = model default)
    enable_profiling: bool = False
    cache_dir: str = "./cache"
    performance_hint: str = "THROUGHPUT"  # THROUGHPUT or LATENCY
    inference_precision: str = "FP32"
    bind_thread: bool = True
    threads_per_stream: int = 0
    enable_hyper_threading: bool = True
    # Device-specific properties for accelerator (passed via -p/--properties CLI)
    device_properties: Dict[str, str] = field(default_factory=dict)

    def get_device_prefix(self) -> str:
        """Get the device prefix (e.g., 'NPU' from 'NPU.0', 'NPU', or 'NPU.0,NPU.2')."""
        device = self.device.upper()
        # Handle comma-separated: "NPU.0,NPU.2" → take first entry
        if "," in device:
            first = device.split(",")[0].strip()
            if "." in first:
                return first.split(".")[0]
            return first
        if "." in device:
            return device.split(".")[0]
        return device

    def is_accelerator_device(self) -> bool:
        """Check if the configured device is a multi-die accelerator (not CPU/GPU)."""
        prefix = self.get_device_prefix()
        return prefix not in ("CPU", "GPU", "AUTO", "MULTI", "HETERO")

    def is_multi_device(self) -> bool:
        """Check if using ALL dies (device without die number like 'NPU', not 'NPU.0' or 'NPU.0,NPU.2')."""
        device = self.device.upper()
        if not self.is_accelerator_device():
            return False
        return "." not in device and "," not in device

    def is_specific_die(self) -> bool:
        """Check if a single specific die is selected (e.g., 'NPU.0')."""
        device = self.device.upper()
        if not self.is_accelerator_device():
            return False
        if "," in device:
            return False
        if "." not in device:
            return False
        suffix = device.split(".", 1)[1]
        return suffix.isdigit()

    def is_selected_dies(self) -> bool:
        """Check if multiple specific dies are selected (e.g., 'NPU.0,NPU.2')."""
        device = self.device.upper()
        if not self.is_accelerator_device():
            return False
        if "," not in device:
            return False
        parts = [p.strip() for p in device.split(",")]
        for p in parts:
            if "." not in p:
                return False
            suffix = p.split(".", 1)[1]
            if not suffix.isdigit():
                return False
        return len(parts) >= 2

    def get_target_devices(self) -> Optional[List[str]]:
        """Get list of target devices for die selection.

        Returns:
            None if all dies should be used (bare prefix like 'NPU').
            List of specific device names otherwise (e.g., ['NPU.0'] or ['NPU.0', 'NPU.2']).
        """
        if self.is_multi_device():
            return None
        device = self.device.upper()
        if self.is_selected_dies():
            return [p.strip() for p in device.split(",")]
        if self.is_specific_die():
            return [device]
        return None


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
    # In-flight request multiplier: controls queue depth
    # Lower = less latency (for Server mode), Higher = more throughput (for Offline mode)
    nireq_multiplier: int = 2
    # Explicit batching (Intel-style) for Server mode
    explicit_batching: bool = False
    explicit_batch_size: int = 8
    batch_timeout_us: int = 2000


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    task: str
    model_type: ModelType
    input_shape: List[int]
    input_name: str = "input"
    output_name: str = "output"
    data_format: str = "NCHW"
    dtype: str = "FP32"

    accuracy_target: float = 0.0
    accuracy_threshold: float = 0.99

    # SDXL-specific: range-based accuracy metrics (clip_score_min/max, fid_score_min/max)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    offline: ScenarioConfig = field(default_factory=ScenarioConfig)
    server: ScenarioConfig = field(default_factory=ScenarioConfig)

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

    mlperf_version: str = "5.1"
    division: str = "open"
    category: str = "datacenter"

    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="ResNet50",
        task="image_classification",
        model_type=ModelType.RESNET50,
        input_shape=[1, 3, 224, 224]
    ))

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="imagenet2012",
        path="./data/imagenet"
    ))

    openvino: OpenVINOConfig = field(default_factory=OpenVINOConfig)

    scenario: Scenario = Scenario.OFFLINE
    test_mode: TestMode = TestMode.PERFORMANCE_ONLY

    results_dir: str = "./results"
    logs_dir: str = "./logs"

    @classmethod
    def from_yaml(cls, yaml_path: str, model_name: str = "resnet50") -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        global_settings = data.get("global", {})

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
            target_qps=offline_data.get("target_qps", 0.0),
        )

        server_data = model_data.get("server", {})
        server = ScenarioConfig(
            min_duration_ms=server_data.get("min_duration_ms", 60000),
            min_query_count=server_data.get("min_query_count", 24576),
            target_latency_ns=server_data.get("target_latency_ns", 15000000),
            target_qps=server_data.get("target_qps", 10000.0),
            qsl_rng_seed=server_data.get("qsl_rng_seed", 13281865557512327830),
            sample_index_rng_seed=server_data.get("sample_index_rng_seed", 198141574272810017),
            schedule_rng_seed=server_data.get("schedule_rng_seed", 7575108116881280410),
            nireq_multiplier=server_data.get("nireq_multiplier", 6),
            explicit_batching=server_data.get("explicit_batching", False),
            explicit_batch_size=server_data.get("explicit_batch_size", 8),
            batch_timeout_us=server_data.get("batch_timeout_us", 2000),
        )

        sources = model_data.get("sources", {})

        model_config = ModelConfig(
            name=model_data.get("name", model_name),
            task=model_data.get("task", "image_classification"),
            model_type=ModelType(model_name),
            input_shape=model_data.get("input_shape", [1, 3, 224, 224]),
            input_name=model_data.get("input_name", "input"),
            output_name=model_data.get("output_name", "output"),
            data_format=model_data.get("data_format", "NCHW"),
            dtype=model_data.get("dtype", "FP32"),
            accuracy_target=model_data.get("accuracy_target", 0.0),
            accuracy_threshold=model_data.get("accuracy_threshold", 0.99),
            preprocessing=preprocessing,
            offline=offline,
            server=server,
            onnx_url=sources.get("onnx_url"),
            accuracy_metrics=model_data.get("accuracy_metrics", {}),
        )

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
            inference_precision=ov_data.get("inference_precision", "FP32"),
            bind_thread=cpu_data.get("bind_thread", True),
            threads_per_stream=cpu_data.get("threads_per_stream", 0),
            enable_hyper_threading=cpu_data.get("enable_hyper_threading", True),
        )

        datasets_data = data.get("datasets", {})
        dataset_name = model_data.get("dataset", "imagenet2012")
        dataset_data = datasets_data.get(dataset_name, {})

        dataset_config = DatasetConfig(
            name=dataset_name,
            path=dataset_data.get("path", f"./data/{dataset_name}"),
            val_map=dataset_data.get("val_map"),
            calibration_path=dataset_data.get("calibration_path"),
        )

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
        """Create default ResNet50 configuration with NPU-optimized Server mode."""
        return cls(
            model=ModelConfig(
                name="ResNet50-v1.5",
                task="image_classification",
                model_type=ModelType.RESNET50,
                input_shape=[1, 3, 224, 224],
                input_name="input",
                output_name="output",
                data_format="NCHW",
                dtype="FP32",
                accuracy_target=0.7646,
                accuracy_threshold=0.99,
                onnx_url="https://zenodo.org/record/4735647/files/resnet50_v1.onnx",
                offline=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=24576,
                    samples_per_query=1,
                    target_qps=6150.0,
                ),
                server=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=24576,
                    target_latency_ns=15000000,  # 15ms (MLPerf official)
                    target_qps=5700.0,
                    nireq_multiplier=6,
                    explicit_batching=True,
                    explicit_batch_size=8,
                    batch_timeout_us=2000,
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
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=1633,  # MLPerf official
                    samples_per_query=1,
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
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=10833,  # MLPerf official (SQuAD dataset size)
                    samples_per_query=1,
                    target_qps=395.0,
                ),
                server=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=10833,
                    target_latency_ns=130000000,  # 130ms (MLPerf official)
                    target_qps=330.0,
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
                input_shape=[1, 3, 800, 800],  # (batch, channels, height, width)
                input_name="input",
                output_name="output",
                data_format="NCHW",
                dtype="FP32",
                accuracy_target=0.3757,  # mAP (official MLPerf v5.1)
                accuracy_threshold=0.99,
                preprocessing=PreprocessingConfig(
                    resize=(800, 800),
                    center_crop=(800, 800),
                    # NOTE: MLPerf RetinaNet uses only /255.0 normalization, NO ImageNet mean/std
                    mean=(0.0, 0.0, 0.0),
                    std=(255.0, 255.0, 255.0),  # Equivalent to /255.0
                    channel_order="RGB",
                    # output_layout="NHWC" (default) - model uses PrePostProcessor for NHWC->NCHW
                ),
                offline=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=24576,  # MLPerf official
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=24576,
                    target_latency_ns=100000000,  # 100ms (MLPerf official)
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
                name="Stable-Diffusion-XL",
                task="text_to_image",
                model_type=ModelType.SDXL,
                input_shape=[1, 77],  # (batch, max_token_length) for text input
                input_name="input_ids",
                output_name="sample",
                data_format="NC",
                dtype="FP32",
                # MLPerf v5.1 accuracy targets for SDXL (closed division)
                # CLIP_SCORE: >= 31.68632 and <= 31.81332
                # FID_SCORE: >= 23.01086 and <= 23.95007
                accuracy_target=31.68632,  # Minimum CLIP score
                accuracy_threshold=1.0,
                accuracy_metrics={
                    'clip_score_min': 31.68632,
                    'clip_score_max': 31.81332,
                    'fid_score_min': 23.01086,
                    'fid_score_max': 23.95007,
                },
                preprocessing=PreprocessingConfig(),  # Not used for text input
                offline=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=5000,  # MLPerf official
                    samples_per_query=1,
                ),
                server=ScenarioConfig(
                    min_duration_ms=600000,  # MLPerf official: 10 minutes
                    min_query_count=5000,  # MLPerf official
                    target_latency_ns=20000000000,  # 20 seconds for image generation
                    target_qps=10.0,
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

        if self.model.model_path and not Path(self.model.model_path).exists():
            errors.append(f"Model file not found: {self.model.model_path}")

        if not Path(self.dataset.path).exists():
            errors.append(f"Dataset path not found: {self.dataset.path}")

        if self.scenario not in [Scenario.OFFLINE, Scenario.SERVER]:
            errors.append(f"Scenario {self.scenario} is not supported in this version")

        supported = SUPPORTED_SCENARIOS.get(self.model.model_type)
        if supported and self.scenario not in supported:
            names = ", ".join(s.value for s in supported)
            errors.append(
                f"{self.model.name} only supports {names} scenario per MLCommons specification"
            )

        return errors
