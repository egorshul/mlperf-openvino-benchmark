# MLPerf OpenVINO Benchmark

MLPerf Inference benchmark implementation using OpenVINO backend.

## Supported Models

| Model | Task | Dataset | Scenario |
|-------|------|---------|----------|
| ResNet50-v1.5 | Image Classification | ImageNet 2012 | Offline, Server |
| BERT-Large | Question Answering | SQuAD v1.1 | Offline, Server |
| RetinaNet | Object Detection | OpenImages | Offline, Server |
| Whisper Large v3 | Speech Recognition | LibriSpeech | Offline, Server |
| Stable Diffusion XL | Text-to-Image | COCO 2014 | Offline, Server |

## Installation

```bash
pip install -e ".[all]"
```

To build the optional C++ SUT for higher performance (bypasses Python GIL):

```bash
./build_cpp.sh
```

## Quick Start

```bash
# Download model and dataset
mlperf-ov setup --model resnet50

# Run accuracy test
mlperf-ov run --model resnet50 --mode accuracy

# Run performance test
mlperf-ov run --model resnet50 --mode performance --scenario Offline
```

## Device Selection

```bash
# CPU (default)
mlperf-ov run --model resnet50 --device CPU

# All accelerator dies (auto-discovery)
mlperf-ov run --model resnet50 --device NPU

# Specific die
mlperf-ov run --model resnet50 --device NPU.0
```

## Run Examples

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

### RetinaNet

```bash
mlperf-ov run --model retinanet --scenario Offline --device NPU
```

### Whisper Large v3

```bash
mlperf-ov run --model whisper --scenario Offline --device CPU
```

### Stable Diffusion XL

```bash
mlperf-ov run --model sdxl --scenario Offline --device CPU
```

## CLI Reference

```
mlperf-ov run                  Run benchmark
mlperf-ov setup --model X      Download model + dataset
mlperf-ov download-model       Download model only
mlperf-ov download-dataset     Download dataset only
mlperf-ov list-models          List supported models
mlperf-ov info                 System information
```

### Options

```
--model, -m      Model: resnet50, bert, retinanet, whisper, sdxl
--mode           Mode: accuracy, performance
--scenario, -s   Scenario: Offline, Server
--device, -d     Device: CPU, NPU, NPU.0, etc.
--batch-size, -b Inference batch size
--count          Number of samples (0 = all)
-p, --properties Device properties (KEY=VALUE,KEY2=VALUE2)
```

## Project Structure

```
src/mlperf_openvino/
    backends/          Inference backends (OpenVINO, multi-device)
    core/              SUT implementations, benchmark runner, config
    datasets/          Dataset loaders and QSL implementations
    utils/             Model and dataset download utilities
    cpp/               C++ SUT implementations (pybind11)
    cli.py             CLI entry point
```

## License

See [LICENSE](LICENSE).
