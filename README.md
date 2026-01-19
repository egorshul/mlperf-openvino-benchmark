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
| **Stable Diffusion XL** | Text-to-Image | COCO 2014 | CLIP Score / FID | 31.69-31.81 / 23.01-23.95 | Within range |

## Installation

```bash
git clone https://github.com/egorshul/mlperf-openvino-benchmark.git
cd mlperf-openvino-benchmark
pip install -e ".[all]"
```

### Additional dependencies for SDXL

```bash
pip install diffusers optimum[openvino] transformers accelerate
```

### Build C++ SUT (Optional, for better performance)

```bash
./build_cpp.sh
```

Requires: CMake 3.14+, C++17 compiler, pybind11

## Quick Start

### Setup model and dataset

```bash
# Classic vision/language models
mlperf-ov setup --model resnet50
mlperf-ov setup --model bert
mlperf-ov setup --model retinanet
mlperf-ov setup --model whisper

# Text-to-Image (SDXL)
mlperf-ov setup --model sdxl --format openvino
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

### SDXL Benchmark

```bash
# Download model and dataset
mlperf-ov download --model sdxl --format openvino
mlperf-ov download-dataset --dataset coco2014

# Run benchmark
mlperf-ov run --model sdxl \
  --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
  --data-path ./data/coco2014 \
  --mode accuracy

# Performance test
mlperf-ov run --model sdxl \
  --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
  --data-path ./data/coco2014 \
  --mode performance \
  --scenario Offline
```

## CLI Reference

### Main Commands

```bash
mlperf-ov run                  # Run benchmark
mlperf-ov download             # Download model
mlperf-ov download-dataset     # Download dataset
mlperf-ov setup                # Download model + dataset
mlperf-ov info                 # System information
mlperf-ov list-models          # List supported models
mlperf-ov perf-tips            # Performance optimization tips
```

### Run Options

```
mlperf-ov run [OPTIONS]

  --model, -m          Model [resnet50|bert|retinanet|whisper|sdxl]
  --scenario, -s       Scenario [Offline|Server]
  --mode               Mode [accuracy|performance|both]
  --model-path         Path to model file
  --data-path          Path to dataset
  --num-threads        CPU threads (0 = auto)
  --num-streams        Inference streams (AUTO or number)
  --batch-size, -b     Batch size
  --count              Number of samples (0 = all)
  --performance-hint   Hint [THROUGHPUT|LATENCY|AUTO]
```

### Download Dataset Options

```
mlperf-ov download-dataset [OPTIONS]

  --dataset, -d        Dataset [imagenet|squad|openimages|librispeech|coco2014]
  --output-dir, -o     Output directory (default: ./data)
  --subset, -s         Subset (e.g., "dev-clean" for librispeech)
```

## Model-Specific Notes

### Stable Diffusion XL (SDXL)

SDXL is a text-to-image generation model. The benchmark:

- Uses 5000 prompts from COCO 2014 captions
- Generates 1024x1024 images
- Evaluates using CLIP Score and FID metrics

**MLPerf v5.1 accuracy targets:**
- CLIP Score: 31.68632 - 31.81332
- FID Score: 23.01086 - 23.95007

**Requirements:**
- ~12GB RAM for model
- Optimum-Intel with OpenVINO support
- diffusers library

**Model components:**
- UNet (main denoising network)
- VAE decoder (latent to image)
- Two CLIP text encoders

#### Manual Download and Conversion (if automatic download fails)

If you encounter network errors (HTTPSConnectionPool, MaxRetryError), follow these manual steps:

**Option 1: Using huggingface-cli (recommended)**

```bash
# Install huggingface CLI
pip install huggingface_hub

# Login (optional, for faster downloads)
huggingface-cli login

# Download the model
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir ./models/stable-diffusion-xl-base-1.0 \
  --local-dir-use-symlinks False

# Convert to OpenVINO format
optimum-cli export openvino \
  --model ./models/stable-diffusion-xl-base-1.0 \
  --task stable-diffusion-xl \
  ./models/stable-diffusion-xl-base-1.0-openvino
```

**Option 2: Using git lfs**

```bash
# Install git-lfs
sudo apt-get install git-lfs  # or: brew install git-lfs
git lfs install

# Clone the model repository
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 \
  ./models/stable-diffusion-xl-base-1.0

# Convert to OpenVINO
optimum-cli export openvino \
  --model ./models/stable-diffusion-xl-base-1.0 \
  --task stable-diffusion-xl \
  ./models/stable-diffusion-xl-base-1.0-openvino
```

**Option 3: Python script for manual conversion**

```python
from optimum.intel import OVStableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline
import torch

# Load PyTorch model (downloads if not cached)
pytorch_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
)

# Export to OpenVINO
ov_pipeline = OVStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    export=True,
    compile=False,
)
ov_pipeline.save_pretrained("./models/stable-diffusion-xl-base-1.0-openvino")
```

**Network troubleshooting tips:**

```bash
# Increase timeout for slow connections
export HF_HUB_DOWNLOAD_TIMEOUT=3600

# Use mirror (if available in your region)
export HF_ENDPOINT=https://hf-mirror.com

# Resume interrupted downloads
export HF_HUB_ENABLE_HF_TRANSFER=0

# Disable SSL verification (use with caution)
export CURL_CA_BUNDLE=""
```

**Expected directory structure after conversion:**

```
models/stable-diffusion-xl-base-1.0-openvino/
├── model_index.json
├── scheduler/
├── text_encoder/
│   └── openvino_model.xml, openvino_model.bin
├── text_encoder_2/
│   └── openvino_model.xml, openvino_model.bin
├── tokenizer/
├── tokenizer_2/
├── unet/
│   └── openvino_model.xml, openvino_model.bin
└── vae_decoder/
    └── openvino_model.xml, openvino_model.bin
```

### Whisper Large v3

Speech recognition model using LibriSpeech dataset:
- Evaluates Word Error Rate (WER) and Word Accuracy
- Uses dev-clean + dev-other subsets

### RetinaNet

Object detection on OpenImages:
- 264 MLPerf classes
- Evaluates using mAP metric

## Performance Tips

For maximum throughput:

```bash
mlperf-ov run --model <model> \
  --scenario Offline \
  --performance-hint THROUGHPUT \
  --batch-size <optimal_batch> \
  --num-streams AUTO
```

Recommended batch sizes:
- ResNet50: 32
- BERT: 8
- RetinaNet: 4
- Whisper: 2
- SDXL: 1 (memory-intensive)

## Project Structure

```
mlperf-openvino-benchmark/
├── src/mlperf_openvino/
│   ├── cli.py                  # Command-line interface
│   ├── backends/               # OpenVINO backend
│   ├── core/                   # Benchmark runner, SUTs
│   │   ├── benchmark_runner.py
│   │   ├── sut.py              # Generic SUT
│   │   ├── bert_sut.py
│   │   ├── retinanet_sut.py
│   │   ├── whisper_sut.py
│   │   └── sdxl_sut.py         # SDXL SUT
│   ├── datasets/               # Dataset loaders
│   │   ├── imagenet.py
│   │   ├── squad.py
│   │   ├── openimages.py
│   │   ├── librispeech.py
│   │   └── coco_prompts.py     # COCO for SDXL
│   └── utils/                  # Downloaders
├── configs/                    # Configuration files
├── tests/                      # Test suite
└── README.md
```

## License

Apache License 2.0
