# MLPerf v5.1 OpenVINO Benchmark

A benchmark tool for measuring CPU inference performance using OpenVINO backend, compatible with MLPerf Inference v5.1 specifications.

## Features

- **MLPerf v5.1 Compatible**: Follows MLPerf Inference benchmark specifications
- **OpenVINO Backend**: Optimized for Intel CPUs using OpenVINO runtime
- **C++ Accelerated SUT**: High-performance C++ implementation bypassing Python GIL
- **Multiple Scenarios**: Supports Offline and Server scenarios
- **Multiple Models**: ResNet50, BERT-Large, RetinaNet, Whisper Large v3
- **Automated Setup**: Built-in model and dataset downloaders

### Supported Models

| Model | Task | Dataset | Metric | Target |
|-------|------|---------|--------|--------|
| ResNet50-v1.5 | Image Classification | ImageNet 2012 | Top-1 Accuracy | 76.46% |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 Score | 90.874% |
| RetinaNet | Object Detection | OpenImages | mAP | 37.57% |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Accuracy | 97.93% |

### Supported Scenarios

- **Offline**: Maximum throughput with all samples available upfront
- **Server**: Latency-constrained throughput with real-time query arrival

## Installation

### Prerequisites

- Python 3.9+
- Intel CPU (recommended: with AVX-512 support)

### Install from source

```bash
git clone https://github.com/egorshul/mlperf-openvino-benchmark.git
cd mlperf-openvino-benchmark
pip install -e .
```

### Install with model-specific dependencies

```bash
# For ResNet50
pip install -e ".[resnet]"

# For BERT
pip install -e ".[bert]"

# For RetinaNet
pip install -e ".[retinanet]"

# For Whisper
pip install -e ".[whisper]"

# All models + development tools
pip install -e ".[all]"
```

### Build C++ SUT Extension (Recommended)

For maximum performance, build the C++ SUT extension which bypasses Python GIL limitations:

```bash
# Build the C++ extension
./build_cpp.sh

# Verify installation
python -c "from mlperf_openvino.cpp import CPP_AVAILABLE; print(f'C++ SUT: {CPP_AVAILABLE}')"
```

**Requirements for C++ build:**
- CMake 3.14+
- C++17 compiler (GCC 7+ or Clang 5+)
- pybind11 (`pip install pybind11`)

The C++ SUT provides significant performance improvements:
- **Server mode**: Async inference with GIL released, pool of InferRequests
- **Offline mode**: Same async approach for maximum parallelism

## Quick Start

### Option 1: One-command setup (recommended)

The easiest way to get started is using the `setup` command which downloads both the model and dataset:

```bash
# Set up ResNet50 benchmark
mlperf-ov setup --model resnet50

# Set up BERT benchmark
mlperf-ov setup --model bert

# Set up RetinaNet benchmark
mlperf-ov setup --model retinanet

# Set up Whisper benchmark
mlperf-ov setup --model whisper
```

After setup completes, follow the printed instructions to run the benchmark.

### Option 2: Manual setup

#### 1. Download the model

```bash
mlperf-ov download --model resnet50 --output-dir ./models
mlperf-ov download --model bert --output-dir ./models
mlperf-ov download --model retinanet --output-dir ./models
mlperf-ov download --model whisper --output-dir ./models --format openvino
```

#### 2. Download the dataset

```bash
mlperf-ov download-dataset --dataset imagenet --output-dir ./data
mlperf-ov download-dataset --dataset squad --output-dir ./data
mlperf-ov download-dataset --dataset openimages --output-dir ./data
mlperf-ov download-dataset --dataset librispeech --output-dir ./data --subset dev-clean
```

#### 3. Run benchmark

```bash
mlperf-ov run \
  --model resnet50 \
  --scenario Offline \
  --mode performance \
  --model-path ./models/resnet50_v1.onnx \
  --data-path ./data/imagenet
```

## CLI Commands

### List available models

```bash
mlperf-ov list-models
```

### Show system information

```bash
mlperf-ov info
```

### Run benchmark

```bash
# Performance test
mlperf-ov run --model resnet50 --mode performance

# Accuracy test
mlperf-ov run --model bert --mode accuracy

# Both accuracy and performance
mlperf-ov run --model retinanet --mode both

# With custom settings
mlperf-ov run \
  --model whisper \
  --scenario Server \
  --num-threads 8 \
  --performance-hint LATENCY
```

### Quick latency benchmark (without LoadGen)

```bash
mlperf-ov benchmark-latency \
  --model-path ./models/resnet50_v1.onnx \
  --iterations 100
```

## Command Line Options

```
mlperf-ov run [OPTIONS]

Options:
  --model, -m          Model to benchmark [resnet50|bert|retinanet|whisper]
  --scenario, -s       Test scenario [Offline|Server]
  --mode               Test mode [accuracy|performance|both]
  --model-path         Path to model file (ONNX or OpenVINO IR)
  --data-path          Path to dataset directory
  --output-dir, -o     Output directory for results
  --config, -c         Path to YAML configuration file
  --num-threads        Number of CPU threads (0 = auto)
  --num-streams        Number of inference streams
  --performance-hint   Performance hint [THROUGHPUT|LATENCY]
  --duration           Minimum test duration in ms
  --count              Number of samples to use (0 = all)
  --warmup             Number of warmup iterations
  --verbose, -v        Enable verbose output
```

## Configuration File

You can use a YAML configuration file for advanced settings:

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

# Create configuration for any supported model
config = BenchmarkConfig.default_resnet50()  # or default_bert(), default_retinanet(), default_whisper()
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
  "accuracy": {
    "top1_accuracy": 0.7648,
    "correct": 38240,
    "total": 50000
  },
  "device": "CPU",
  "openvino_version": "2024.0.0",
  "timestamp": "2025-01-13T10:30:00"
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
│   ├── core/                    # Core components
│   │   ├── config.py            # Configuration management
│   │   ├── sut.py               # Python System Under Test
│   │   ├── cpp_sut_wrapper.py   # C++ SUT Python wrapper
│   │   ├── bert_sut.py          # BERT System Under Test
│   │   ├── retinanet_sut.py     # RetinaNet System Under Test
│   │   ├── whisper_sut.py       # Whisper System Under Test
│   │   └── benchmark_runner.py  # Main benchmark orchestrator
│   ├── cpp/                     # C++ accelerated components
│   │   ├── sut_cpp.cpp/hpp      # C++ SUT (async, GIL-free)
│   │   ├── offline_sut.cpp/hpp  # C++ Offline SUT (batch)
│   │   └── bindings.cpp         # pybind11 bindings
│   ├── backends/                # Inference backends
│   │   └── openvino_backend.py  # OpenVINO inference
│   ├── datasets/                # Dataset handlers
│   │   ├── imagenet.py          # ImageNet for ResNet50
│   │   ├── squad.py             # SQuAD v1.1 for BERT
│   │   ├── openimages.py        # OpenImages for RetinaNet
│   │   └── librispeech.py       # LibriSpeech for Whisper
│   ├── utils/                   # Utilities
│   │   ├── dataset_downloader.py
│   │   └── model_downloader.py
│   └── cli.py                   # Command line interface
├── configs/                     # Configuration files
├── tests/                       # Unit tests
└── results/                     # Benchmark results
```

### C++ SUT Architecture

The C++ SUT bypasses Python GIL for maximum inference throughput:

```
┌─────────────────────────────────────────────────────────────┐
│                     Python (LoadGen)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ issue_query │───▶│ CppSUT      │───▶│ QuerySamples    │ │
│  │             │    │ Wrapper     │    │ Complete        │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────┘ │
└────────────────────────────┼────────────────────────────────┘
                             │ GIL Released
┌────────────────────────────▼────────────────────────────────┐
│                        C++ SUT                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ InferRequest│    │ InferRequest│    │ InferRequest    │ │
│  │ Pool (async)│    │ Pool (async)│    │ Pool (async)    │ │
│  └──────┬──────┘    └──────┬──────┘    └───────┬─────────┘ │
│         │                  │                    │           │
│         └──────────────────┼────────────────────┘           │
│                            ▼                                │
│                    OpenVINO Runtime                         │
└─────────────────────────────────────────────────────────────┘
```

**Key benefits:**
- Inference callbacks run without Python GIL
- Pool of async InferRequests for parallel execution
- Minimal Python ↔ C++ data copying

## MLPerf Compliance

This benchmark follows MLPerf Inference v5.1 specifications:

- Uses official MLPerf LoadGen library for query generation
- Implements QSL (Query Sample Library) and SUT (System Under Test) interfaces
- Supports both Closed and Open division requirements
- Follows accuracy and latency constraints

### Accuracy Requirements (MLPerf v5.1)

| Model | Metric | Reference | Target (99%) |
|-------|--------|-----------|--------------|
| ResNet50-v1.5 | Top-1 Accuracy | 76.46% | >= 75.70% |
| BERT-Large | F1 Score | 90.874% | >= 89.97% |
| RetinaNet | mAP | 37.57% | >= 37.19% |
| Whisper Large v3 | Word Accuracy | 97.93% | >= 96.95% |

### Scenario Requirements

| Scenario | Metric | ResNet50 | BERT | RetinaNet | Whisper |
|----------|--------|----------|------|-----------|---------|
| Offline | Min Duration | 60s | 60s | 60s | 60s |
| Offline | Min Queries | 24,576 | 10,833 | 24,576 | 2,513 |
| Server | Target Latency | 15ms | 130ms | 100ms | 1,000ms |

## Performance Optimization

### Understanding Scenarios

**IMPORTANT**: For maximum throughput, use **Offline** mode, not Server mode!

| Scenario | Purpose | Throughput | Use Case |
|----------|---------|------------|----------|
| **Offline** | Maximum throughput | Unlimited | Batch processing, benchmarking peak performance |
| **Server** | Latency-constrained | Rate-limited | Real-time services, latency testing |

Server mode intentionally limits throughput to test latency compliance. If your Server test is running slowly, that's by design!

### Performance Tips Command

```bash
# Show performance optimization tips for your model
mlperf-ov perf-tips --model resnet50
```

### Quick Throughput Benchmark

Test your hardware's maximum inference speed without LoadGen overhead:

```bash
mlperf-ov benchmark-throughput \
  --model-path ./models/resnet50_v1.onnx \
  --batch-size 32 \
  --iterations 200
```

### Optimal Settings for Each Scenario

**Offline (Maximum Throughput):**

```bash
mlperf-ov run \
  --model resnet50 \
  --scenario Offline \
  --mode performance \
  --batch-size 32 \
  --num-streams AUTO \
  --performance-hint THROUGHPUT \
  --model-path ./models/resnet50_v1.onnx \
  --data-path ./data/imagenet
```

**Server (Low Latency + High Throughput):**

```bash
mlperf-ov run \
  --model resnet50 \
  --scenario Server \
  --mode performance \
  --batch-size 1 \
  --num-streams AUTO \
  --performance-hint THROUGHPUT \
  --target-qps 1000 \
  --model-path ./models/resnet50_v1.onnx \
  --data-path ./data/imagenet
```

Key insight: `num_streams=AUTO` with `THROUGHPUT` hint enables maximum parallelism. `batch_size=1` ensures each query is processed immediately (required by Server mode).

**AUTO mode (recommended):**

The CLI automatically selects optimal settings based on scenario:

```bash
# Automatically uses THROUGHPUT + batch_size=32 + streams=AUTO for Offline
mlperf-ov run --model resnet50 --scenario Offline

# Automatically uses THROUGHPUT + batch_size=1 + streams=AUTO for Server
mlperf-ov run --model resnet50 --scenario Server
```

Both scenarios use `THROUGHPUT` hint for maximum parallelism. The key difference is batch size.

### Key Optimization Parameters

| Parameter | For Throughput (Offline) | For Server |
|-----------|--------------------------|--------------------------|
| `--batch-size` | 32-64 (larger batches) | 1 (single sample per query) |
| `--num-streams` | `AUTO` (parallelism) | `AUTO` (parallelism) |
| `--performance-hint` | `THROUGHPUT` | `THROUGHPUT` |
| `--num-threads` | 0 (auto) | 0 (auto) |

**Important:** Both scenarios use `THROUGHPUT` hint for parallelism. `LATENCY` hint uses only 1 stream which severely limits throughput.

### Recommended Batch Sizes

| Model | Recommended Batch Size | Notes |
|-------|----------------------|-------|
| ResNet50 | 32-64 | Small model, high parallelism |
| BERT | 8-16 | Larger model, moderate batching |
| RetinaNet | 4-8 | Large model, memory-limited |
| Whisper | 1-4 | Sequential decoding limits batching |

## Troubleshooting

### Common Issues

**Model not found error:**
```bash
# Download the model first
mlperf-ov download --model <model_name>
```

**Dataset not found error:**
```bash
# Download the dataset first
mlperf-ov download-dataset --dataset <dataset_name>
```

**OpenVINO not installed:**
```bash
pip install openvino>=2024.0.0
```

**MLPerf LoadGen not installed:**
```bash
pip install mlcommons-loadgen>=4.0
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- [MLCommons](https://mlcommons.org/) for MLPerf benchmark specifications
- [Intel OpenVINO](https://docs.openvino.ai/) for the inference runtime
