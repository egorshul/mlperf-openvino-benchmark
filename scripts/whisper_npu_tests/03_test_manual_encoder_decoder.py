#!/usr/bin/env python3
"""
Step 3: Test manual encoder-decoder inference on NPU.

If Optimum Intel doesn't work with NPU, we need to use OpenVINO
directly. This script tests loading encoder and decoder separately.

Whisper model structure (after optimum export):
    model_path/
    ├── encoder_model.xml / .bin
    ├── decoder_model.xml / .bin           # First token (no KV-cache)
    ├── decoder_with_past_model.xml / .bin # Subsequent tokens (with KV-cache)
    └── config.json, tokenizer files, etc.

Usage:
    python 03_test_manual_encoder_decoder.py --model-path /path/to/whisper/openvino

Requirements:
    pip install openvino transformers numpy
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_model_files(model_path: Path) -> Dict[str, Optional[Path]]:
    """Find encoder and decoder model files."""
    print("\n[1/5] Looking for model files...")

    files = {
        "encoder": None,
        "decoder": None,
        "decoder_with_past": None,
    }

    # Patterns to look for
    patterns = {
        "encoder": ["encoder_model.xml", "encoder.xml"],
        "decoder": ["decoder_model.xml", "decoder.xml"],
        "decoder_with_past": ["decoder_with_past_model.xml", "decoder_with_past.xml"],
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            candidate = model_path / pattern
            if candidate.exists():
                files[key] = candidate
                print(f"  ✓ Found {key}: {candidate.name}")
                break

        if files[key] is None:
            print(f"  ✗ Not found: {key}")

    return files


def load_models_on_device(
    model_files: Dict[str, Path],
    device: str
) -> Dict[str, any]:
    """Load models on specified device using OpenVINO Core."""
    print(f"\n[2/5] Loading models on {device}...")

    import openvino as ov

    core = ov.Core()
    models = {}

    # Configuration for NPU
    config = {}

    # Try different configurations
    configs_to_try = [
        {},
        {"PERFORMANCE_HINT": "LATENCY"},
        {"PERFORMANCE_HINT": "THROUGHPUT"},
    ]

    for name, path in model_files.items():
        if path is None:
            continue

        print(f"\n  Loading {name}...")

        loaded = False
        for config in configs_to_try:
            try:
                start_time = time.time()

                # Read model
                model = core.read_model(str(path))

                # Print model info
                print(f"    Inputs: {[inp.get_any_name() for inp in model.inputs]}")
                print(f"    Outputs: {[out.get_any_name() for out in model.outputs]}")

                # Compile for device
                compiled = core.compile_model(model, device, config)

                load_time = time.time() - start_time
                print(f"    ✓ Compiled in {load_time:.2f}s with config: {config}")

                models[name] = {
                    "model": model,
                    "compiled": compiled,
                    "config": config,
                }
                loaded = True
                break

            except Exception as e:
                print(f"    ✗ Failed with config {config}: {e}")

        if not loaded:
            print(f"    ✗ Could not load {name} on {device}")

    return models


def analyze_model_io(models: Dict[str, any]) -> Dict[str, Dict]:
    """Analyze input/output tensor shapes and names."""
    print("\n[3/5] Analyzing model I/O...")

    io_info = {}

    for name, model_data in models.items():
        compiled = model_data["compiled"]

        inputs = {}
        for inp in compiled.inputs:
            inp_name = inp.get_any_name()
            inp_shape = inp.get_partial_shape()
            inp_dtype = inp.get_element_type()
            inputs[inp_name] = {
                "shape": str(inp_shape),
                "dtype": str(inp_dtype),
            }

        outputs = {}
        for out in compiled.outputs:
            out_name = out.get_any_name()
            out_shape = out.get_partial_shape()
            out_dtype = out.get_element_type()
            outputs[out_name] = {
                "shape": str(out_shape),
                "dtype": str(out_dtype),
            }

        io_info[name] = {
            "inputs": inputs,
            "outputs": outputs,
        }

        print(f"\n  [{name}]")
        print(f"    Inputs:")
        for inp_name, info in inputs.items():
            print(f"      {inp_name}: {info['shape']} ({info['dtype']})")
        print(f"    Outputs:")
        for out_name, info in outputs.items():
            print(f"      {out_name}: {info['shape']} ({info['dtype']})")

    return io_info


def test_encoder_inference(
    models: Dict[str, any],
    batch_size: int = 1
) -> Tuple[bool, Optional[np.ndarray], float]:
    """Test encoder inference with dummy input."""
    print("\n[4/5] Testing encoder inference...")

    if "encoder" not in models:
        print("  ✗ Encoder not loaded")
        return False, None, 0

    compiled = models["encoder"]["compiled"]

    # Create dummy mel spectrogram input
    # Whisper expects: (batch, n_mels, time_frames)
    # n_mels = 80 or 128 depending on model version
    # time_frames = 3000 for 30 seconds of audio

    # Detect expected shape from model
    input_info = compiled.inputs[0]
    input_shape = input_info.get_partial_shape()

    # Try to get static shape or use defaults
    try:
        n_mels = input_shape[1].get_length()
        time_frames = input_shape[2].get_length()
    except Exception:
        n_mels = 128  # Whisper large v3 uses 128 mels
        time_frames = 3000

    print(f"  Input shape: ({batch_size}, {n_mels}, {time_frames})")

    # Create dummy input
    dummy_input = np.random.randn(batch_size, n_mels, time_frames).astype(np.float32)

    # Run inference
    try:
        start_time = time.time()
        result = compiled({compiled.inputs[0]: dummy_input})
        inference_time = time.time() - start_time

        # Get output
        output = result[compiled.outputs[0]]
        print(f"  ✓ Encoder inference successful!")
        print(f"    Output shape: {output.shape}")
        print(f"    Inference time: {inference_time*1000:.1f}ms")

        return True, output, inference_time

    except Exception as e:
        print(f"  ✗ Encoder inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0


def test_decoder_inference(
    models: Dict[str, any],
    encoder_output: np.ndarray
) -> Tuple[bool, float]:
    """Test decoder inference with encoder output."""
    print("\n[5/5] Testing decoder inference...")

    # Prefer decoder_with_past for efficiency, fallback to decoder
    decoder_key = "decoder_with_past" if "decoder_with_past" in models else "decoder"

    if decoder_key not in models:
        print("  ✗ No decoder loaded")
        return False, 0

    compiled = models[decoder_key]["compiled"]

    # Prepare inputs
    batch_size = encoder_output.shape[0]
    encoder_seq_len = encoder_output.shape[1]

    # Whisper special tokens
    SOT_TOKEN = 50258  # Start of transcript
    EN_TOKEN = 50259   # English
    TRANSCRIBE_TOKEN = 50359  # Transcribe task
    NO_TIMESTAMPS_TOKEN = 50363  # No timestamps

    # Initial decoder input: [SOT, language, task, no_timestamps]
    initial_tokens = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]
    decoder_input_ids = np.array([initial_tokens], dtype=np.int64)

    print(f"  Decoder: {decoder_key}")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Initial decoder input: {decoder_input_ids.shape}")

    # Build input dict based on model's expected inputs
    input_names = [inp.get_any_name() for inp in compiled.inputs]
    print(f"  Expected inputs: {input_names}")

    inputs = {}

    # Map inputs
    for inp_name in input_names:
        name_lower = inp_name.lower()

        if "input_id" in name_lower or "decoder_input" in name_lower:
            inputs[inp_name] = decoder_input_ids
        elif "encoder_hidden" in name_lower or "encoder_output" in name_lower:
            inputs[inp_name] = encoder_output
        elif "encoder_attention_mask" in name_lower:
            # All ones for encoder attention mask
            mask = np.ones((batch_size, encoder_seq_len), dtype=np.int64)
            inputs[inp_name] = mask
        elif "attention_mask" in name_lower and "encoder" not in name_lower:
            # Decoder attention mask
            mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[inp_name] = mask
        elif "past_key_value" in name_lower or "cache" in name_lower:
            # KV cache - need to provide empty or zeros
            # Shape depends on model - this is complex
            print(f"    Warning: KV cache input '{inp_name}' needs proper handling")

    print(f"  Prepared inputs: {list(inputs.keys())}")

    # Try inference
    try:
        start_time = time.time()
        result = compiled(inputs)
        inference_time = time.time() - start_time

        # Get output (logits)
        output = result[compiled.outputs[0]]
        print(f"  ✓ Decoder inference successful!")
        print(f"    Output shape: {output.shape}")
        print(f"    Inference time: {inference_time*1000:.1f}ms")

        # Get predicted token
        if output.ndim == 3:
            logits = output[0, -1, :]  # Last token logits
        else:
            logits = output[0, :]

        next_token = int(np.argmax(logits))
        print(f"    Next token ID: {next_token}")

        return True, inference_time

    except Exception as e:
        print(f"  ✗ Decoder inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description="Test manual encoder-decoder inference on NPU"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to OpenVINO Whisper model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="NPU",
        help="Device to use (NPU, NPU.0, etc.)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Manual Encoder-Decoder NPU Test")
    print("=" * 60)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n✗ Model path not found: {model_path}")
        return 1

    # Find model files
    model_files = find_model_files(model_path)

    if model_files["encoder"] is None:
        print("\n✗ Encoder model not found!")
        print("  Make sure you exported with optimum-cli:")
        print("  optimum-cli export openvino --model openai/whisper-large-v3 \\")
        print("    --task automatic-speech-recognition-with-past ./models/whisper")
        return 1

    # Load models
    models = load_models_on_device(model_files, args.device)

    if not models:
        print(f"\n✗ Could not load any models on {args.device}")
        return 1

    # Analyze I/O
    io_info = analyze_model_io(models)

    # Test encoder
    encoder_ok, encoder_output, encoder_time = test_encoder_inference(models)

    if not encoder_ok:
        print("\n" + "=" * 60)
        print("Result: FAILED - Encoder doesn't work on NPU")
        print("=" * 60)
        return 1

    # Test decoder
    decoder_ok, decoder_time = test_decoder_inference(models, encoder_output)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Encoder on {args.device}: {'✓ OK' if encoder_ok else '✗ FAIL'}")
    print(f"  Decoder on {args.device}: {'✓ OK' if decoder_ok else '✗ FAIL'}")

    if encoder_ok:
        print(f"  Encoder time: {encoder_time*1000:.1f}ms")
    if decoder_ok:
        print(f"  Decoder step time: {decoder_time*1000:.1f}ms")

    if encoder_ok and decoder_ok:
        print("\n" + "=" * 60)
        print("Result: SUCCESS! Manual inference works on NPU!")
        print("=" * 60)
        print("\nNext step: Run 04_test_multi_die.py for multi-die support")
        return 0
    elif encoder_ok:
        print("\n" + "=" * 60)
        print("Result: PARTIAL - Encoder works, decoder needs work")
        print("=" * 60)
        print("\nThe decoder may need:")
        print("  1. Proper KV-cache handling")
        print("  2. Different input preparation")
        print("  3. Model re-export with different options")
        return 1
    else:
        print("\n" + "=" * 60)
        print("Result: FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
