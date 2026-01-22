# MLPerf v5.1 OpenVINO Benchmark

MLPerf Inference v5.1 benchmark implementation using OpenVINO backend.

## Supported Models

| Model | Task | Dataset | Metric | Reference | Required (99%) |
|-------|------|---------|--------|-----------|----------------|
| ResNet50 | Image Classification | ImageNet 2012 | Top-1 Accuracy | 76.46% | ≥ 75.70% |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 Score | 90.87% | ≥ 89.97% |
| RetinaNet | Object Detection | OpenImages | mAP | 37.57% | ≥ 37.19% |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Accuracy | 97.93% | ≥ 96.95% |
| SDXL | Text-to-Image | COCO 2014 | CLIP / FID | 31.69 / 23.01 | 31.68-31.81 / 23.01-23.95 |

## Installation

```bash
git clone https://github.com/egorshul/mlperf-openvino-benchmark.git
cd mlperf-openvino-benchmark

# Basic installation
pip install -e .

# With specific model dependencies
pip install -e ".[resnet]"     # ResNet50
pip install -e ".[bert]"       # BERT
pip install -e ".[retinanet]"  # RetinaNet
pip install -e ".[whisper]"    # Whisper
pip install -e ".[sdxl]"       # Stable Diffusion XL

# All dependencies
pip install -e ".[all]"
```

### C++ SUT (Optional, for better performance)

```bash
# Requirements: CMake 3.14+, C++17 compiler, pybind11
./build_cpp.sh

# Verify installation
python -c "from mlperf_openvino.cpp import CPP_AVAILABLE; print(f'C++ SUT: {CPP_AVAILABLE}')"
```

## Quick Start

### 1. Download Model and Dataset

```bash
# ResNet50
mlperf-ov download-model --model resnet50
mlperf-ov download-dataset --dataset imagenet

# BERT
mlperf-ov download-model --model bert
mlperf-ov download-dataset --dataset squad

# RetinaNet
mlperf-ov download-model --model retinanet
mlperf-ov download-dataset --dataset openimages

# Whisper
mlperf-ov download-model --model whisper --format openvino
mlperf-ov download-dataset --dataset librispeech

# SDXL
mlperf-ov download-model --model sdxl --format openvino
mlperf-ov download-dataset --dataset coco2014
```

### 2. Run Benchmark

```bash
# Accuracy test
mlperf-ov run --model resnet50 --mode accuracy

# Performance test (Offline scenario)
mlperf-ov run --model resnet50 --mode performance --scenario Offline

# Performance test (Server scenario)
mlperf-ov run --model resnet50 --mode performance --scenario Server
```

## CLI Reference

```bash
mlperf-ov run                  # Run benchmark
mlperf-ov download-model       # Download model
mlperf-ov download-dataset     # Download dataset
mlperf-ov setup                # Download model + dataset
mlperf-ov list-models          # List supported models
mlperf-ov info                 # System information
```

### Run Options

```
--model, -m          Model: resnet50, bert, retinanet, whisper, sdxl
--mode               Mode: accuracy, performance
--scenario, -s       Scenario: Offline, Server
--device, -d         Device: CPU, NPU, NPU.0, NPU.1, etc.
--model-path         Path to model
--data-path          Path to dataset
--count              Number of samples (0 = all)
--batch-size, -b     Batch size
--num-threads        CPU threads (0 = auto)
--num-streams        Inference streams
```

### Server Mode Options

```
--target-qps              Target queries per second
--target-latency-ns       Target latency in nanoseconds (default: 15ms for ResNet50)
--nireq-multiplier        In-flight requests multiplier (default: 2)
--explicit-batching       Enable Intel-style explicit batching (recommended for NPU)
--batch-timeout-us        Batch timeout in microseconds (default: 500)
```

## NPU/Accelerator Support

For multi-die NPU accelerators (e.g., Intel NPU with multiple tiles):

### Offline Mode

```bash
mlperf-ov run --model resnet50 --scenario Offline \
  --device NPU \
  --batch-size 8
```

### Server Mode (Optimized)

For maximum throughput with latency constraints:

```bash
mlperf-ov run --model resnet50 --scenario Server \
  --device NPU \
  --target-qps 5750 \
  --explicit-batching \
  --batch-size 8 \
  --batch-timeout-us 2000 \
  --nireq-multiplier 6
```

### Performance Tuning

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `--batch-size` | Samples per inference | Higher = better throughput |
| `--batch-timeout-us` | Max wait before flush | Lower = lower latency |
| `--nireq-multiplier` | In-flight requests | Higher = better utilization |
| `--target-qps` | Target throughput | Start low, increase until INVALID |

**Tuning strategy:**
1. Start with `--explicit-batching -b 8 --batch-timeout-us 1000 --nireq-multiplier 2`
2. Run with low `--target-qps`, verify VALID
3. Increase `--target-qps` until test becomes INVALID
4. Tune `--nireq-multiplier` (2-6) and `--batch-timeout-us` (500-2000)
