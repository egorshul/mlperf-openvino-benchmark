# MLPerf v5.1 OpenVINO Benchmark

MLPerf Inference v5.1 benchmark implementation using OpenVINO backend for CPU.

## Supported Models

| Model | Task | Dataset | Metric | Target |
|-------|------|---------|--------|--------|
| ResNet50 | Image Classification | ImageNet 2012 | Top-1 Accuracy | 76.46% |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 Score | 90.87% |
| RetinaNet | Object Detection | OpenImages | mAP | 37.57% |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Accuracy | 97.93% |
| SDXL | Text-to-Image | COCO 2014 | CLIP / FID | 31.7 / 23.5 |

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
mlperf-ov download --model resnet50
mlperf-ov download-dataset --dataset imagenet

# BERT
mlperf-ov download --model bert
mlperf-ov download-dataset --dataset squad

# RetinaNet
mlperf-ov download --model retinanet
mlperf-ov download-dataset --dataset openimages

# Whisper
mlperf-ov download --model whisper --format openvino
mlperf-ov download-dataset --dataset librispeech

# SDXL
mlperf-ov download --model sdxl --format openvino
mlperf-ov download-dataset --dataset coco2014
```

### 2. Run Benchmark

```bash
# Accuracy test
mlperf-ov run --model resnet50 --mode accuracy

# Performance test
mlperf-ov run --model resnet50 --mode performance --scenario Offline

# Both accuracy and performance
mlperf-ov run --model resnet50 --mode both
```

## CLI Reference

```bash
mlperf-ov run                  # Run benchmark
mlperf-ov download             # Download model
mlperf-ov download-dataset     # Download dataset
mlperf-ov list-models          # List supported models
mlperf-ov info                 # System information
```

### Run Options

```
--model, -m          Model: resnet50, bert, retinanet, whisper, sdxl
--mode               Mode: accuracy, performance, both
--scenario, -s       Scenario: Offline, Server
--model-path         Path to model
--data-path          Path to dataset
--count              Number of samples (0 = all)
--batch-size, -b     Batch size
--num-threads        CPU threads (0 = auto)
--num-streams        Inference streams
```

## License

Apache License 2.0
