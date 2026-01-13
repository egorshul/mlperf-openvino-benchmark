# MLPerf v5.1 OpenVINO Benchmark

A benchmark tool for measuring CPU inference performance using OpenVINO backend, compatible with MLPerf Inference v5.1 specifications.

## Features

- **MLPerf v5.1 Compatible**: Follows MLPerf Inference benchmark specifications
- **OpenVINO Backend**: Optimized for Intel CPUs using OpenVINO runtime
- **Multiple Scenarios**: Supports Offline and Server scenarios
- **Extensible Architecture**: Designed to easily add new models

### Supported Models

| Model | Task | Status |
|-------|------|--------|
| ResNet50 | Image Classification | ✅ Implemented |
| BERT-Large | Question Answering | 🔜 Planned |
| RetinaNet | Object Detection | 🔜 Planned |
| Whisper | Speech to Text | 🔜 Planned |
| Stable Diffusion XL | Text to Image | 🔜 Planned |

### Supported Scenarios

- **Offline**: Maximum throughput with all samples available upfront
- **Server**: Latency-constrained throughput with real-time query arrival

## Installation

### Prerequisites

- Python 3.9+
- Intel CPU (recommended: with AVX-512 support)

### Install from source

```bash
git clone https://github.com/your-org/mlperf-openvino-benchmark.git
cd mlperf-openvino-benchmark
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

### Install all optional dependencies

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Download the model

```bash
mlperf-ov download --model resnet50 --output-dir ./models
```

### 2. Prepare the dataset

Download ImageNet 2012 validation dataset and create the val_map.txt file:

```bash
# After downloading ImageNet validation set
ls /path/to/imagenet/val/*.JPEG > val_list.txt
# Create val_map.txt with format: filename label
```

### 3. Run benchmark

```bash
# Performance test
mlperf-ov run \
  --model resnet50 \
  --scenario Offline \
  --mode performance \
  --model-path ./models/resnet50_v1.onnx \
  --data-path ./data/imagenet

# Accuracy test
mlperf-ov run \
  --model resnet50 \
  --scenario Offline \
  --mode accuracy \
  --model-path ./models/resnet50_v1.onnx \
  --data-path ./data/imagenet
```

### Quick latency benchmark (without LoadGen)

```bash
mlperf-ov benchmark-latency \
  --model-path ./models/resnet50_v1.onnx \
  --iterations 100
```

## Configuration

### Command Line Options

```
Options:
  --model, -m          Model to benchmark [resnet50|bert|retinanet]
  --scenario, -s       Test scenario [Offline|Server]
  --mode               Test mode [accuracy|performance|both]
  --model-path         Path to model file (ONNX or OpenVINO IR)
  --data-path          Path to dataset directory
  --output-dir, -o     Output directory for results
  --num-threads        Number of CPU threads (0 = auto)
  --num-streams        Number of inference streams
  --performance-hint   Performance hint [THROUGHPUT|LATENCY]
  --duration           Minimum test duration in ms
```

### Configuration File

You can also use a YAML configuration file:

```yaml
# benchmark_config.yaml
global:
  mlperf_version: "5.1"
  division: "open"

models:
  resnet50:
    name: "ResNet50-v1.5"
    task: "image_classification"
    input_shape: [1, 3, 224, 224]
    accuracy_target: 0.7646

openvino:
  device: "CPU"
  num_streams: "AUTO"
  performance_hint: "THROUGHPUT"

datasets:
  imagenet2012:
    path: "./data/imagenet"
    val_map: "./data/imagenet/val_map.txt"
```

Run with config file:

```bash
mlperf-ov run --config benchmark_config.yaml --model resnet50
```

## Python API

```python
from mlperf_openvino import BenchmarkRunner, BenchmarkConfig
from mlperf_openvino.core import Scenario, TestMode

# Create configuration
config = BenchmarkConfig.default_resnet50()
config.model.model_path = "./models/resnet50_v1.onnx"
config.dataset.path = "./data/imagenet"
config.scenario = Scenario.OFFLINE
config.test_mode = TestMode.PERFORMANCE_ONLY

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Print summary
runner.print_summary()

# Save results
runner.save_results("./results/benchmark_results.json")
```

## Results

Results are saved in JSON format:

```json
{
  "model": "ResNet50-v1.5",
  "scenario": "Offline",
  "mode": "PerformanceOnly",
  "duration_seconds": 60.5,
  "samples_processed": 50000,
  "throughput_samples_per_sec": 826.45,
  "device": "CPU",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Development

### Running tests

```bash
pytest tests/ -v
```

### Code formatting

```bash
black src/
ruff check src/
```

### Type checking

```bash
mypy src/
```

## Architecture

```
mlperf-openvino-benchmark/
├── src/mlperf_openvino/
│   ├── core/           # Core components
│   │   ├── config.py   # Configuration management
│   │   ├── sut.py      # System Under Test
│   │   └── benchmark_runner.py
│   ├── backends/       # Inference backends
│   │   └── openvino_backend.py
│   ├── datasets/       # Dataset handlers
│   │   └── imagenet.py
│   ├── utils/          # Utilities
│   └── cli.py          # Command line interface
├── configs/            # Configuration files
├── tests/              # Unit tests
└── results/            # Benchmark results
```

## MLPerf Compliance

This benchmark follows MLPerf Inference v5.1 specifications:

- Uses official MLPerf LoadGen library for query generation
- Implements QSL (Query Sample Library) and SUT (System Under Test) interfaces
- Supports both Closed and Open division requirements
- Follows accuracy and latency constraints

### Accuracy Requirements

| Model | Metric | Target |
|-------|--------|--------|
| ResNet50 | Top-1 Accuracy | ≥ 75.70% (99% of 76.46%) |
| BERT | F1 Score | ≥ 89.18% (99% of 90.08%) |

### Scenario Requirements

| Scenario | Metric | Constraint |
|----------|--------|------------|
| Offline | Throughput | Maximum samples/sec |
| Server | Latency | 99th percentile ≤ 15ms (ResNet50) |

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- [MLCommons](https://mlcommons.org/) for MLPerf benchmark specifications
- [Intel OpenVINO](https://docs.openvino.ai/) for the inference runtime
