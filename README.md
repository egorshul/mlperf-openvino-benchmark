# MLPerf OpenVINO Benchmark v1.0.0

MLPerf Inference v5.1 benchmark implementation using Intel OpenVINO as the inference backend. Supports multi-die accelerators (NPU/VPU) with automatic device discovery and both Python and C++ SUT implementations.

## Supported Models

| Model | Task | Dataset | Metric | Target (closed, 99%) | Scenarios |
|-------|------|---------|--------|-----------------------|-----------|
| ResNet50-v1.5 | Image Classification | ImageNet 2012 | Top-1 Accuracy | >= 75.70% | Offline, Server |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 Score | >= 89.97% | Offline, Server |
| RetinaNet | Object Detection | OpenImages | mAP | >= 37.19% | Offline, Server |
| SSD-ResNet34 | Object Detection | COCO 2017 | mAP | >= 19.80% | Offline, Server |
| 3D UNET | Medical Image Segmentation | KiTS 2019 | Mean DICE | >= 85.47% | Offline |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Accuracy | >= 96.95% | Offline |
| Stable Diffusion XL | Text-to-Image | COCO 2014 | CLIP / FID | 31.69-31.81 / 23.01-23.95 | Offline, Server |

## Requirements

- Python >= 3.9
- OpenVINO >= 2024.0.0
- MLCommons LoadGen >= 4.0

## Installation

Install the package with dependencies for all models:

```bash
pip install -e ".[all]"
```

Or install only the dependencies you need:

```bash
# Core only (ResNet50)
pip install -e .

# Specific model
pip install -e ".[bert]"
pip install -e ".[whisper]"
pip install -e ".[sdxl]"
pip install -e ".[retinanet]"
pip install -e ".[ssd-resnet34]"
pip install -e ".[3d-unet]"

# Development
pip install -e ".[dev]"
```

### Optional: C++ SUT

For maximum Server mode throughput, build the C++ SUT extension. This bypasses the Python GIL and calls `mlperf::QuerySamplesComplete` directly from C++.

```bash
./build_cpp.sh
```

Requires CMake 3.14+, a C++17 compiler, and pybind11.

Verify:

```bash
python -c "from mlperf_openvino.cpp import ResNetCppSUT, CPP_AVAILABLE; print(f'C++ SUT available: {CPP_AVAILABLE}')"
```

## Quick Start

```bash
# Download model and dataset
mlperf-ov setup --model resnet50

# Run accuracy test
mlperf-ov run --model resnet50 --mode accuracy

# Run performance test
mlperf-ov run --model resnet50 --mode performance --scenario Offline

# Run both
mlperf-ov run --model resnet50 --mode both --scenario Offline
```

## Device Selection

```bash
# CPU (default)
mlperf-ov run --model resnet50 --device CPU

# All accelerator dies (auto-discovery)
mlperf-ov run --model resnet50 --device NPU

# Specific die
mlperf-ov run --model resnet50 --device NPU.0

# Multiple specific dies
mlperf-ov run --model resnet50 --device NPU.0,NPU.2
```

When targeting an accelerator (e.g. `NPU`), the framework automatically discovers all available dies and distributes inference across them using a thread pool with round-robin dispatch.

## Model Examples

### ResNet50

```bash
mlperf-ov setup --model resnet50
mlperf-ov run --model resnet50 --scenario Offline --device NPU
mlperf-ov run --model resnet50 --scenario Server --device NPU --explicit-batching
```

### BERT-Large

```bash
mlperf-ov setup --model bert
mlperf-ov run --model bert --scenario Offline --device NPU
mlperf-ov run --model bert --scenario Server --device NPU
```

### RetinaNet

```bash
mlperf-ov setup --model retinanet
mlperf-ov run --model retinanet --scenario Offline --device NPU
mlperf-ov run --model retinanet --scenario Server --device NPU
```

### SSD-ResNet34

```bash
mlperf-ov setup --model ssd-resnet34
mlperf-ov run --model ssd-resnet34 --scenario Offline --device CPU
mlperf-ov run --model ssd-resnet34 --scenario Offline --device NPU
mlperf-ov run --model ssd-resnet34 --scenario Server --device NPU
```

### 3D UNET

Medical image segmentation on the KiTS 2019 kidney tumor dataset. Uses sliding window inference with 128x128x128 sub-volumes and Gaussian-weighted aggregation. Only supports Offline scenario per MLPerf specification.

```bash
mlperf-ov setup --model 3d-unet
mlperf-ov run --model 3d-unet --scenario Offline --device CPU
mlperf-ov run --model 3d-unet --scenario Offline --device NPU
mlperf-ov run --model 3d-unet --mode accuracy --device CPU
```

### Whisper Large v3

```bash
mlperf-ov setup --model whisper
mlperf-ov run --model whisper --scenario Offline --device CPU
mlperf-ov run --model whisper --scenario Offline --device NPU
```

### Stable Diffusion XL

SDXL is automatically downloaded in OpenVINO IR format with FP32 weights.

```bash
mlperf-ov setup --model sdxl
mlperf-ov run --model sdxl --scenario Offline --device CPU
mlperf-ov run --model sdxl --scenario Offline --device NPU
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `mlperf-ov run` | Run benchmark |
| `mlperf-ov setup --model X` | Download model and dataset |
| `mlperf-ov download-model` | Download model only |
| `mlperf-ov download-dataset` | Download dataset only |
| `mlperf-ov list-models` | List supported models |
| `mlperf-ov info` | Show system and library information |

### Run Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model: resnet50, bert, retinanet, ssd-resnet34, 3d-unet, whisper, sdxl | (required) |
| `--mode` | Mode: accuracy, performance, both | accuracy |
| `--scenario, -s` | Scenario: Offline, Server | Offline |
| `--device, -d` | Device: CPU, NPU, NPU.0, NPU.0,NPU.2 | CPU |
| `--model-path` | Path to model (ONNX or OpenVINO IR) | auto |
| `--data-path` | Path to dataset | auto |
| `--config, -c` | Config YAML path | configs/benchmark_config.yaml |
| `--output-dir, -o` | Output directory | ./results |
| `--batch-size, -b` | Inference batch size | 1 |
| `--count` | Number of samples (0 = all) | 0 |
| `--warmup` | Warmup iterations | 10 |
| `--duration` | Test duration (ms) | from config |
| `--target-qps` | Target QPS | from config |
| `--target-latency-ns` | Target latency in ns (Server) | from config |
| `-p, --properties` | Device properties (KEY=VALUE,KEY2=VALUE2) | - |
| `--num-threads` | Number of threads (0 = auto) | 0 |
| `--num-streams` | Number of inference streams | AUTO |
| `--performance-hint` | THROUGHPUT, LATENCY, AUTO | AUTO |
| `--nchw` | Use NCHW layout (ResNet50) | NHWC |
| `--explicit-batching` | Enable explicit batching (Server) | false |
| `--verbose, -v` | Enable verbose output | false |

### Download Options

```bash
# Download model in specific format
mlperf-ov download-model --model resnet50 --format onnx
mlperf-ov download-model --model resnet50 --format openvino

# Download specific dataset
mlperf-ov download-dataset --dataset imagenet
mlperf-ov download-dataset --dataset librispeech --subset dev-clean
mlperf-ov download-dataset --dataset coco2017
mlperf-ov download-dataset --dataset kits19
mlperf-ov download-dataset --dataset coco2014 --with-images

# Download RetinaNet with batch sizes
mlperf-ov download-model --model retinanet --batch-sizes 1,2,4,8
```

## Configuration

The main configuration is in `configs/benchmark_config.yaml`. It includes:

- MLPerf version, division (closed/open), and category (datacenter/edge)
- Per-model settings: input shapes, accuracy targets, preprocessing, scenario parameters
- OpenVINO settings: device, streams, threads, precision, performance hints
- Dataset paths
- LoadGen settings

Key OpenVINO parameters:

```yaml
openvino:
  device: "CPU"
  num_streams: "AUTO"
  num_threads: 0
  batch_size: 1
  performance_hint: "THROUGHPUT"
  inference_precision: "FP32"
```

## Architecture

### Multi-Die Inference

The framework auto-discovers accelerator dies and distributes inference across all of them:

```
LoadGen
  |
  v
SUT (Python / C++)
  |
  +--> Die 0  (NPU.0)  ─── inference ───> results
  +--> Die 1  (NPU.1)  ─── inference ───> results
  +--> Die 2  (NPU.2)  ─── inference ───> results
  +--> Die 3  (NPU.3)  ─── inference ───> results
  |
  v
QuerySamplesComplete
```

1 card = 2 dies in NPU topology. The number of dies is not limited.

### C++ SUT (Server Mode)

For Server mode, the C++ SUT calls `mlperf::QuerySamplesComplete` directly from C++ without acquiring the Python GIL:

| Mode | Expected QPS (ResNet50) |
|------|-------------------------|
| Python | ~17,000 |
| C++ | ~55,000 |

### Project Structure

```
mlperf-openvino-benchmark/
├── configs/
│   └── benchmark_config.yaml     # Main configuration
├── src/mlperf_openvino/
│   ├── cli.py                    # CLI entry point (mlperf-ov)
│   ├── backends/                 # Inference backends
│   │   ├── openvino_backend.py   # Single-device OpenVINO backend
│   │   ├── multi_device_backend.py # Multi-die backend
│   │   └── device_discovery.py   # Accelerator die detection
│   ├── core/                     # SUT implementations and runner
│   │   ├── config.py             # Configuration dataclasses
│   │   ├── benchmark_runner.py   # Main benchmark orchestrator
│   │   ├── sut_factory.py        # Factory for model-specific SUTs
│   │   ├── sut.py                # Generic OpenVINO SUT
│   │   ├── resnet_multi_die_sut.py
│   │   ├── bert_multi_die_sut.py
│   │   ├── retinanet_multi_die_sut.py
│   │   ├── ssd_resnet34_sut.py
│   │   ├── ssd_resnet34_multi_die_sut.py
│   │   ├── unet3d_sut.py
│   │   ├── unet3d_multi_die_sut.py
│   │   ├── whisper_multi_die_sut.py
│   │   ├── sdxl_multi_die_sut.py
│   │   ├── whisper_sut.py
│   │   └── sdxl_sut.py
│   ├── datasets/                 # Dataset loaders and QSL
│   │   ├── imagenet.py           # ImageNet 2012 (ResNet50)
│   │   ├── squad.py              # SQuAD v1.1 (BERT)
│   │   ├── openimages.py         # OpenImages (RetinaNet)
│   │   ├── coco.py               # COCO 2017 (SSD-ResNet34)
│   │   ├── kits19.py             # KiTS 2019 (3D UNET)
│   │   ├── librispeech.py        # LibriSpeech (Whisper)
│   │   └── coco_prompts.py       # COCO 2014 (SDXL)
│   ├── utils/
│   │   ├── model_downloader.py   # Model download from Zenodo/HuggingFace
│   │   └── dataset_downloader.py # Dataset download
│   └── cpp/                      # C++ SUT (pybind11)
│       ├── CMakeLists.txt
│       ├── resnet_multi_die_sut_cpp.hpp/cpp
│       ├── ssd_resnet34_multi_die_sut_cpp.hpp/cpp
│       └── unet3d_multi_die_sut_cpp.hpp/cpp
├── build_cpp.sh                  # Build script for C++ SUT
└── pyproject.toml
```
