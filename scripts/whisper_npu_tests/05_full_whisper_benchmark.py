#!/usr/bin/env python3
"""
Step 5: Full Whisper benchmark on NPU with multi-die support.

This script runs complete Whisper inference (encoder + decoder + text generation)
on NPU dies. It's a prototype for the final WhisperMultiDeviceSUT.

Architecture:
    Sample 0 → NPU.0: encode → decode → text
    Sample 1 → NPU.1: encode → decode → text
    Sample 2 → NPU.0: encode → decode → text
    ...

Usage:
    python 05_full_whisper_benchmark.py --model-path /path/to/whisper/openvino \\
        --audio /path/to/test.wav

Or with dummy audio:
    python 05_full_whisper_benchmark.py --model-path /path/to/whisper/openvino \\
        --num-samples 8

Requirements:
    pip install openvino transformers numpy soundfile
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Whisper special tokens
SOT_TOKEN = 50258       # Start of transcript
EOT_TOKEN = 50257       # End of transcript
TRANSCRIBE_TOKEN = 50359    # Transcribe task
NO_TIMESTAMPS_TOKEN = 50363  # No timestamps
EN_TOKEN = 50259        # English language


@dataclass
class WhisperDieContext:
    """Context for Whisper inference on a single die."""
    device_name: str
    compiled_encoder: any
    compiled_decoder: any  # decoder_with_past or decoder
    decoder_input_names: Dict[str, str] = field(default_factory=dict)
    decoder_has_past: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Statistics
    encode_count: int = 0
    decode_steps: int = 0
    total_encode_time: float = 0.0
    total_decode_time: float = 0.0


def discover_npu_dies() -> List[str]:
    """Discover all available NPU dies."""
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    return sorted([d for d in devices if d.startswith("NPU.") and d[4:].isdigit()])


def load_whisper_on_dies(
    model_path: Path,
    dies: List[str]
) -> Dict[str, WhisperDieContext]:
    """Load Whisper encoder and decoder on all dies."""
    import openvino as ov

    core = ov.Core()
    contexts = {}

    # Find model files
    encoder_path = model_path / "encoder_model.xml"
    decoder_path = model_path / "decoder_with_past_model.xml"

    if not encoder_path.exists():
        encoder_path = model_path / "encoder.xml"
    if not decoder_path.exists():
        decoder_path = model_path / "decoder_model.xml"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found at {model_path}")

    print(f"  Encoder: {encoder_path.name}")
    print(f"  Decoder: {decoder_path.name if decoder_path.exists() else 'NOT FOUND'}")

    # Read models once
    encoder_model = core.read_model(str(encoder_path))

    decoder_model = None
    has_past = "with_past" in str(decoder_path)
    if decoder_path.exists():
        decoder_model = core.read_model(str(decoder_path))

    # Compile on each die
    for die in dies:
        print(f"\n  Compiling on {die}...")

        try:
            # Compile encoder
            start = time.time()
            compiled_encoder = core.compile_model(encoder_model, die)
            print(f"    Encoder: ✓ ({time.time()-start:.1f}s)")

            # Compile decoder
            compiled_decoder = None
            if decoder_model:
                start = time.time()
                compiled_decoder = core.compile_model(decoder_model, die)
                print(f"    Decoder: ✓ ({time.time()-start:.1f}s)")

            # Discover decoder input names
            decoder_input_names = {}
            if compiled_decoder:
                for inp in compiled_decoder.inputs:
                    name = inp.get_any_name()
                    name_lower = name.lower()
                    if "input_id" in name_lower or "decoder_input" in name_lower:
                        decoder_input_names["input_ids"] = name
                    elif "encoder_hidden" in name_lower or "encoder_output" in name_lower:
                        decoder_input_names["encoder_hidden_states"] = name
                    elif "encoder_attention_mask" in name_lower:
                        decoder_input_names["encoder_attention_mask"] = name
                    elif "attention_mask" in name_lower:
                        decoder_input_names["attention_mask"] = name

            contexts[die] = WhisperDieContext(
                device_name=die,
                compiled_encoder=compiled_encoder,
                compiled_decoder=compiled_decoder,
                decoder_input_names=decoder_input_names,
                decoder_has_past=has_past,
            )

        except Exception as e:
            print(f"    ✗ Failed: {e}")

    return contexts


def encode_audio(
    context: WhisperDieContext,
    mel_input: np.ndarray
) -> np.ndarray:
    """Run encoder on mel spectrogram."""
    compiled = context.compiled_encoder

    start = time.time()
    result = compiled({compiled.inputs[0]: mel_input})
    encode_time = time.time() - start

    context.encode_count += 1
    context.total_encode_time += encode_time

    return result[compiled.outputs[0]]


def decode_step(
    context: WhisperDieContext,
    encoder_hidden_states: np.ndarray,
    decoder_input_ids: np.ndarray
) -> np.ndarray:
    """Run one decoder step."""
    if context.compiled_decoder is None:
        raise RuntimeError("Decoder not loaded")

    compiled = context.compiled_decoder
    input_names = context.decoder_input_names

    # Prepare inputs
    inputs = {}

    if "input_ids" in input_names:
        inputs[input_names["input_ids"]] = decoder_input_ids.astype(np.int64)
    if "encoder_hidden_states" in input_names:
        inputs[input_names["encoder_hidden_states"]] = encoder_hidden_states
    if "encoder_attention_mask" in input_names:
        batch_size = encoder_hidden_states.shape[0]
        seq_len = encoder_hidden_states.shape[1]
        inputs[input_names["encoder_attention_mask"]] = np.ones((batch_size, seq_len), dtype=np.int64)
    if "attention_mask" in input_names:
        inputs[input_names["attention_mask"]] = np.ones(decoder_input_ids.shape, dtype=np.int64)

    start = time.time()
    result = compiled(inputs)
    decode_time = time.time() - start

    context.decode_steps += 1
    context.total_decode_time += decode_time

    return result[compiled.outputs[0]]


def generate_text(
    context: WhisperDieContext,
    encoder_hidden_states: np.ndarray,
    max_new_tokens: int = 100,
    tokenizer=None
) -> Tuple[List[int], str]:
    """Generate text tokens autoregressively."""
    if context.compiled_decoder is None:
        return [], "[decoder not available]"

    # Initial tokens: SOT, language, task, no_timestamps
    tokens = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]

    for step in range(max_new_tokens):
        # Prepare input
        decoder_input = np.array([tokens], dtype=np.int64)

        # Get logits
        logits = decode_step(context, encoder_hidden_states, decoder_input)

        # Get next token (greedy)
        if logits.ndim == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[0, :]

        next_token = int(np.argmax(next_token_logits))

        # Check for end
        if next_token == EOT_TOKEN:
            break

        tokens.append(next_token)

    # Decode tokens to text
    generated_tokens = tokens[4:]  # Skip special tokens

    if tokenizer:
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        text = f"[{len(generated_tokens)} tokens]"

    return generated_tokens, text


def process_sample_on_die(
    context: WhisperDieContext,
    mel_input: np.ndarray,
    sample_idx: int,
    max_new_tokens: int = 100,
    tokenizer=None
) -> Tuple[int, str, float, float, int]:
    """Process one sample completely on a die."""
    with context.lock:
        # Encode
        encode_start = time.time()
        encoder_output = encode_audio(context, mel_input)
        encode_time = time.time() - encode_start

        # Generate
        generate_start = time.time()
        tokens, text = generate_text(
            context, encoder_output, max_new_tokens, tokenizer
        )
        generate_time = time.time() - generate_start

    return sample_idx, text, encode_time, generate_time, len(tokens)


def create_dummy_mel(n_mels: int = 128, time_frames: int = 3000) -> np.ndarray:
    """Create dummy mel spectrogram."""
    return np.random.randn(1, n_mels, time_frames).astype(np.float32)


def load_audio_to_mel(audio_path: str) -> np.ndarray:
    """Load audio file and convert to mel spectrogram."""
    try:
        import soundfile as sf
        from transformers import WhisperFeatureExtractor

        # Load audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                ratio = 16000 / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio)-1, new_len),
                    np.arange(len(audio)),
                    audio
                )

        # Extract mel features
        extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        features = extractor(audio, sampling_rate=16000, return_tensors="np")

        return features["input_features"]

    except Exception as e:
        print(f"  Warning: Could not load audio ({e}), using dummy input")
        return create_dummy_mel()


def main():
    parser = argparse.ArgumentParser(
        description="Full Whisper benchmark on NPU"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to OpenVINO Whisper model"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples for benchmark (if no audio)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate per sample"
    )
    parser.add_argument(
        "--dies",
        type=str,
        default=None,
        help="Comma-separated dies to use"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5: Full Whisper NPU Benchmark")
    print("=" * 60)

    # Discover dies
    print("\n[1/5] Discovering NPU dies...")
    available_dies = discover_npu_dies()

    if not available_dies:
        print("  ✗ No NPU dies found!")
        return 1

    if args.dies:
        dies = [d.strip() for d in args.dies.split(",")]
    else:
        dies = available_dies

    print(f"  Using {len(dies)} dies: {dies}")

    # Load models
    print("\n[2/5] Loading Whisper models...")
    model_path = Path(args.model_path)

    try:
        contexts = load_whisper_on_dies(model_path, dies)
    except Exception as e:
        print(f"  ✗ Failed to load models: {e}")
        return 1

    if not contexts:
        print("  ✗ No dies loaded successfully")
        return 1

    print(f"\n  ✓ Loaded on {len(contexts)} dies")

    # Load tokenizer
    print("\n[3/5] Loading tokenizer...")
    tokenizer = None
    try:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
        print("  ✓ Tokenizer loaded")
    except Exception as e:
        print(f"  ✗ Tokenizer not available: {e}")

    # Prepare inputs
    print("\n[4/5] Preparing inputs...")

    if args.audio:
        print(f"  Loading audio: {args.audio}")
        mel_input = load_audio_to_mel(args.audio)
        mel_inputs = [mel_input] * args.num_samples
    else:
        print(f"  Creating {args.num_samples} dummy inputs")
        mel_inputs = [create_dummy_mel() for _ in range(args.num_samples)]

    print(f"  Input shape: {mel_inputs[0].shape}")

    # Run benchmark
    print("\n[5/5] Running benchmark...")
    print(f"  Samples: {len(mel_inputs)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Dies: {len(contexts)}")

    die_names = list(contexts.keys())
    results = []

    total_start = time.time()

    # Parallel processing
    with ThreadPoolExecutor(max_workers=len(contexts)) as executor:
        futures = []

        for i, mel in enumerate(mel_inputs):
            die_idx = i % len(die_names)
            die_name = die_names[die_idx]
            context = contexts[die_name]

            future = executor.submit(
                process_sample_on_die,
                context, mel, i, args.max_tokens, tokenizer
            )
            futures.append((future, die_name))

        for future, die_name in futures:
            try:
                result = future.result()
                results.append((die_name,) + result)
            except Exception as e:
                print(f"  ✗ Error on {die_name}: {e}")

    total_time = time.time() - total_start

    # Print results
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)

    for die_name, sample_idx, text, enc_time, gen_time, num_tokens in results:
        print(f"\n  Sample {sample_idx} on {die_name}:")
        print(f"    Encode: {enc_time*1000:.1f}ms")
        print(f"    Generate: {gen_time*1000:.1f}ms ({num_tokens} tokens)")
        print(f"    Total: {(enc_time+gen_time)*1000:.1f}ms")
        if text:
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"    Text: '{display_text}'")

    # Statistics
    print("\n" + "-" * 60)
    print("Statistics:")
    print("-" * 60)

    for die_name, ctx in contexts.items():
        print(f"\n  {die_name}:")
        print(f"    Encodes: {ctx.encode_count}")
        print(f"    Decode steps: {ctx.decode_steps}")
        if ctx.encode_count > 0:
            print(f"    Avg encode: {ctx.total_encode_time/ctx.encode_count*1000:.1f}ms")
        if ctx.decode_steps > 0:
            print(f"    Avg decode step: {ctx.total_decode_time/ctx.decode_steps*1000:.1f}ms")

    throughput = len(results) / total_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    if throughput > 0:
        print(f"\n  ✓ Successfully processed {len(results)} samples on {len(contexts)} dies")
        print(f"  ✓ Achieved {throughput:.2f} samples/sec throughput")
        print("\n  Ready for integration into mlperf_openvino library!")
    else:
        print("\n  ✗ Benchmark failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
