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
pip install -e ".[resnet]"    # ResNet50
pip install -e ".[bert]"      # BERT
pip install -e ".[whisper]"   # Whisper
pip install -e ".[sdxl]"      # Stable Diffusion XL

# All dependencies
pip install -e ".[all]"
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

## SDXL Benchmark

SDXL generates 1024x1024 images from text prompts using COCO 2014 captions (5000 samples).

**MLPerf v5.1 accuracy targets:**
- CLIP Score: 31.68 - 31.81
- FID Score: 23.01 - 23.95

```bash
# Download and setup
mlperf-ov download --model sdxl --format openvino
mlperf-ov download-dataset --dataset coco2014

# Run benchmark
mlperf-ov run --model sdxl \
  --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
  --data-path ./data/coco2014 \
  --mode accuracy
```

**Manual model download** (if automatic fails):

```bash
# Option 1: Using optimum-cli
pip install optimum[openvino]
optimum-cli export openvino \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  ./models/stable-diffusion-xl-base-1.0-openvino

# Option 2: Using huggingface-cli
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir ./models/stable-diffusion-xl-base-1.0
optimum-cli export openvino \
  --model ./models/stable-diffusion-xl-base-1.0 \
  ./models/stable-diffusion-xl-base-1.0-openvino
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

## Performance Tips

```bash
mlperf-ov run --model <model> \
  --scenario Offline \
  --performance-hint THROUGHPUT \
  --batch-size <batch> \
  --num-streams AUTO
```

Recommended batch sizes:
- ResNet50: 32
- BERT: 8
- RetinaNet: 4
- Whisper: 2
- SDXL: 1

## Project Structure

```
src/mlperf_openvino/
├── cli.py                 # CLI interface
├── core/
│   ├── benchmark_runner.py
│   ├── sdxl_sut.py       # SDXL implementation
│   └── ...
├── datasets/
│   ├── coco_prompts.py   # COCO for SDXL
│   └── ...
└── utils/
    ├── dataset_downloader.py
    └── model_downloader.py
```

## License

Apache License 2.0
