#!/usr/bin/env python3
"""
Step 2: Check if Optimum Intel supports NPU for Whisper.

This script attempts to load Whisper model using optimum-intel
with NPU device. This is the simplest path if it works.

Usage:
    python 02_check_optimum_npu.py [--model-path /path/to/whisper/openvino]

Requirements:
    pip install optimum[openvino] transformers torch
"""

import argparse
import sys
import time
from pathlib import Path


def check_optimum_installation():
    """Check if optimum-intel is installed."""
    print("\n[1/4] Checking optimum-intel installation...")

    try:
        import optimum
        try:
            from optimum.version import __version__ as optimum_version
        except ImportError:
            optimum_version = "unknown"
        print(f"  ✓ optimum version: {optimum_version}")
    except ImportError:
        print("  ✗ optimum not installed")
        print("    Install with: pip install optimum[openvino]")
        return False

    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        print("  ✓ OVModelForSpeechSeq2Seq available")
        return True
    except ImportError as e:
        print(f"  ✗ OVModelForSpeechSeq2Seq not available: {e}")
        print("    Install with: pip install optimum[openvino]")
        return False


def check_transformers():
    """Check transformers installation."""
    print("\n[2/4] Checking transformers installation...")

    try:
        import transformers
        print(f"  ✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers not installed")
        print("    Install with: pip install transformers")
        return False

    try:
        from transformers import AutoProcessor
        print("  ✓ AutoProcessor available")
        return True
    except ImportError as e:
        print(f"  ✗ AutoProcessor not available: {e}")
        return False


def try_load_model_on_npu(model_path: str, device: str = "NPU"):
    """Try to load Whisper model on NPU using optimum-intel."""
    print(f"\n[3/4] Attempting to load model on {device}...")

    from optimum.intel.openvino import OVModelForSpeechSeq2Seq

    # Try different device configurations
    configs_to_try = [
        # Config 1: Direct device specification
        {"device": device},
        # Config 2: With cache disabled
        {"device": device, "ov_config": {"CACHE_DIR": ""}},
        # Config 3: With performance hint
        {"device": device, "ov_config": {"PERFORMANCE_HINT": "THROUGHPUT"}},
    ]

    model = None
    successful_config = None

    for i, config in enumerate(configs_to_try):
        print(f"\n  Trying config {i+1}: {config}")
        try:
            start_time = time.time()
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                compile=True,
                **config
            )
            load_time = time.time() - start_time
            print(f"  ✓ Model loaded successfully in {load_time:.2f}s!")
            successful_config = config
            break
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")

    return model, successful_config


def try_inference(model, model_path: str):
    """Try to run inference with the loaded model."""
    print("\n[4/4] Testing inference...")

    import numpy as np
    import torch
    from transformers import AutoProcessor

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        print("  Loading processor from openai/whisper-large-v3...")
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    # Create dummy input (30 seconds of silence)
    print("  Creating dummy audio input...")
    sample_rate = 16000
    duration = 5  # 5 seconds for quick test
    dummy_audio = np.zeros(sample_rate * duration, dtype=np.float32)

    # Add some noise to avoid silence detection
    dummy_audio += np.random.randn(len(dummy_audio)).astype(np.float32) * 0.01

    # Preprocess
    print("  Preprocessing audio...")
    inputs = processor(
        dummy_audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    input_features = inputs.input_features

    print(f"  Input shape: {input_features.shape}")

    # Run inference
    print("  Running inference...")
    try:
        start_time = time.time()
        generated_ids = model.generate(
            input_features,
            max_new_tokens=50,  # Short for quick test
            language="en",
            task="transcribe",
        )
        inference_time = time.time() - start_time

        # Decode
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"  ✓ Inference successful in {inference_time:.2f}s!")
        print(f"  Generated text: '{text}'")
        return True, inference_time

    except Exception as e:
        print(f"  ✗ Inference failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description="Check if Optimum Intel supports NPU for Whisper"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to OpenVINO Whisper model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="NPU",
        help="Device to use (NPU, NPU.0, etc.)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model from HuggingFace if not found"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Optimum Intel NPU Support Check")
    print("=" * 60)

    # Check dependencies
    if not check_optimum_installation():
        return 1

    if not check_transformers():
        return 1

    # Determine model path
    model_path = args.model_path
    if model_path is None:
        # Try default locations
        default_paths = [
            Path("./models/whisper-large-v3-openvino"),
            Path("./models/whisper"),
            Path("../models/whisper-large-v3-openvino"),
            Path.home() / ".cache/huggingface/hub/models--openai--whisper-large-v3",
        ]
        for p in default_paths:
            if p.exists():
                model_path = str(p)
                print(f"\n  Found model at: {model_path}")
                break

    if model_path is None:
        if args.download:
            print("\n  Downloading model from HuggingFace...")
            model_path = "openai/whisper-large-v3"
        else:
            print("\n✗ No model path specified and no model found!")
            print("  Use --model-path /path/to/model")
            print("  Or use --download to download from HuggingFace")
            print("\n  Hint: Export model to OpenVINO format with:")
            print("  optimum-cli export openvino --model openai/whisper-large-v3 \\")
            print("    --task automatic-speech-recognition-with-past ./models/whisper-large-v3-openvino")
            return 1

    print(f"\n  Using model: {model_path}")

    # Try to load model on NPU
    model, config = try_load_model_on_npu(model_path, args.device)

    if model is None:
        print("\n" + "=" * 60)
        print("Result: Optimum Intel does NOT support NPU directly")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Try with specific die: --device NPU.0")
        print("  2. If still fails, proceed to 03_test_manual_encoder_decoder.py")
        print("     for manual OpenVINO inference")
        return 1

    # Try inference
    success, inference_time = try_inference(model, model_path)

    print("\n" + "=" * 60)
    if success:
        print("Result: SUCCESS! Optimum Intel works with NPU!")
        print("=" * 60)
        print(f"\nConfiguration that worked: {config}")
        print(f"Inference time: {inference_time:.2f}s")
        print("\nYou can use WhisperOptimumSUT with NPU device directly.")
        print("Integration should be straightforward.")
        print("\nNext step: Run 04_test_multi_die.py for multi-die support")
    else:
        print("Result: Model loads but inference fails")
        print("=" * 60)
        print("\nThis might be fixable. Proceed to:")
        print("  03_test_manual_encoder_decoder.py")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
