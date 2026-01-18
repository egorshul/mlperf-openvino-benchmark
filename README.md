# MLPerf v5.1 OpenVINO Benchmark

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![MLPerf](https://img.shields.io/badge/MLPerf-v5.1-green.svg)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.0+-orange.svg)

MLPerf Inference v5.1 benchmark for OpenVINO backend.

## Supported Models & Accuracy Targets

| Model | Task | Dataset | Metric | Reference | Target (99%) |
|-------|------|---------|--------|-----------|--------------|
| ResNet50-v1.5 | Image Classification | ImageNet 2012 | Top-1 Accuracy | 76.46% | ≥ 75.70% |
| BERT-Large | Question Answering | SQuAD v1.1 | F1 Score | 90.874% | ≥ 89.97% |
| RetinaNet | Object Detection | OpenImages | mAP | 37.57% | ≥ 37.19% |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Word Accuracy | 97.93% | ≥ 96.95% |

## Installation

```bash
git clone https://github.com/egorshul/mlperf-openvino-benchmark.git
cd mlperf-openvino-benchmark
pip install -e ".[all]"
```

### Build C++ SUT (Optional, for better performance)

```bash
./build_cpp.sh
```

Requires: CMake 3.14+, C++17 compiler, pybind11

## Quick Start

### Setup model and dataset

```bash
mlperf-ov setup --model resnet50
mlperf-ov setup --model bert
mlperf-ov setup --model retinanet
mlperf-ov setup --model whisper
```

### Run benchmark

```bash
# Accuracy test
mlperf-ov run --model resnet50 --mode accuracy

# Performance test
mlperf-ov run --model resnet50 --mode performance --scenario Offline

# Both
mlperf-ov run --model resnet50 --mode both
```

### CLI Options

```
mlperf-ov run [OPTIONS]

  --model, -m          Model [resnet50|bert|retinanet|whisper]
  --scenario, -s       Scenario [Offline|Server]
  --mode               Mode [accuracy|performance|both]
  --model-path         Path to model file
  --data-path          Path to dataset
  --num-threads        CPU threads (0 = auto)
  --count              Number of samples (0 = all)
```

## License

Apache License 2.0
