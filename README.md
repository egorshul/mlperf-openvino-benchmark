# MLPerf v5.1 OpenVINO Benchmark

MLPerf Inference v5.1 benchmark implementation using OpenVINO backend.

## Supported Models

| Model | Task | Dataset | Metric | Target (99%) |
|-------|------|---------|--------|--------------|
| ResNet50 | Image Classification | ImageNet | Top-1 | ≥ 75.70% |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 | ≥ 89.97% |
| RetinaNet | Object Detection | OpenImages | mAP | ≥ 37.19% |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Acc | ≥ 96.95% |
| SDXL | Text-to-Image | COCO 2014 | CLIP/FID | 31.68-31.81 |
| 3D-UNet | Medical Image Segmentation | KiTS19 | Mean Dice | ≥ 85.47% |

## Installation

```bash
pip install -e ".[all]"

# Optional: C++ SUT for better performance
./build_cpp.sh
```

## Quick Start

```bash
# Setup (download model + dataset)
mlperf-ov setup --model resnet50

# Run benchmark
mlperf-ov run --model resnet50 --mode accuracy
mlperf-ov run --model resnet50 --mode performance --scenario Offline
```

## Optimal Run Commands (NPU)

### ResNet50

```bash
mlperf-ov run --model resnet50 --scenario Offline --device NPU
mlperf-ov run --model resnet50 --scenario Server --device NPU --explicit-batching
```

### BERT-Large

```bash
mlperf-ov run --model bert --scenario Offline --device NPU
mlperf-ov run --model bert --scenario Server --device NPU
```

## Device Selection

```bash
# CPU (default)
mlperf-ov run --model resnet50 --device CPU

# All accelerator dies
mlperf-ov run --model resnet50 --device NPU

# Specific die
mlperf-ov run --model resnet50 --device NPU.0
```

## CLI Reference

```
mlperf-ov run                  # Run benchmark
mlperf-ov setup --model X      # Download model + dataset
mlperf-ov download-model       # Download model only
mlperf-ov download-dataset     # Download dataset only
mlperf-ov list-models          # List supported models
mlperf-ov info                 # System information
```

### Common Options

```
--model, -m      Model: resnet50, bert, retinanet, whisper, sdxl, 3d-unet
--mode           Mode: accuracy, performance
--scenario, -s   Scenario: Offline, Server
--device, -d     Device: CPU, NPU, NPU.0, etc.
--batch-size, -b Batch size
--count          Number of samples (0 = all)
```
