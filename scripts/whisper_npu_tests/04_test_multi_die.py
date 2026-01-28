#!/usr/bin/env python3
"""
Step 4: Test multi-die distribution for Whisper on NPU.

This script tests loading encoder on multiple NPU dies and running
parallel inference. This is key for maximizing throughput.

Strategy:
- Load encoder on each die (NPU.0, NPU.1, etc.)
- Distribute audio samples across dies
- Each die processes its samples independently
- Collect results

Usage:
    python 04_test_multi_die.py --model-path /path/to/whisper/openvino

Requirements:
    pip install openvino numpy
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DieContext:
    """Context for a single NPU die."""
    device_name: str
    compiled_encoder: any
    compiled_decoder: any
    infer_count: int = 0
    total_time: float = 0.0


def discover_npu_dies() -> List[str]:
    """Discover all available NPU dies."""
    import openvino as ov

    core = ov.Core()
    devices = core.available_devices

    # Find specific dies (NPU.0, NPU.1, etc.)
    npu_dies = sorted([d for d in devices if d.startswith("NPU.") and d[4:].isdigit()])

    return npu_dies


def load_encoder_on_dies(
    encoder_path: Path,
    dies: List[str]
) -> Dict[str, DieContext]:
    """Load encoder model on all specified dies."""
    import openvino as ov

    print(f"\nLoading encoder on {len(dies)} dies...")

    core = ov.Core()
    contexts = {}

    # Read model once
    encoder_model = core.read_model(str(encoder_path))

    for die in dies:
        print(f"  Compiling on {die}...", end=" ", flush=True)
        try:
            start = time.time()
            compiled = core.compile_model(encoder_model, die)
            compile_time = time.time() - start
            print(f"✓ ({compile_time:.1f}s)")

            contexts[die] = DieContext(
                device_name=die,
                compiled_encoder=compiled,
                compiled_decoder=None,  # We'll add decoder later if needed
            )
        except Exception as e:
            print(f"✗ ({e})")

    return contexts


def run_encoder_on_die(
    context: DieContext,
    mel_input: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run encoder inference on a specific die."""
    compiled = context.compiled_encoder

    start = time.time()
    result = compiled({compiled.inputs[0]: mel_input})
    inference_time = time.time() - start

    output = result[compiled.outputs[0]]

    context.infer_count += 1
    context.total_time += inference_time

    return output, inference_time


def test_sequential_inference(
    contexts: Dict[str, DieContext],
    num_samples: int = 8
) -> Dict[str, List[float]]:
    """Test sequential inference on each die."""
    print(f"\n[Test 1] Sequential inference ({num_samples} samples per die)...")

    # Create dummy inputs
    # Assuming Whisper large v3: (1, 128, 3000)
    dummy_input = np.random.randn(1, 128, 3000).astype(np.float32)

    results = {}

    for die_name, context in contexts.items():
        print(f"\n  {die_name}:", flush=True)
        times = []

        for i in range(num_samples):
            _, inference_time = run_encoder_on_die(context, dummy_input)
            times.append(inference_time)
            print(f"    Sample {i+1}: {inference_time*1000:.1f}ms")

        results[die_name] = times

        avg_time = np.mean(times[1:]) if len(times) > 1 else times[0]  # Skip warmup
        print(f"    Average (excl. warmup): {avg_time*1000:.1f}ms")

    return results


def test_parallel_inference(
    contexts: Dict[str, DieContext],
    num_samples: int = 16
) -> Tuple[float, int]:
    """Test parallel inference across all dies."""
    num_dies = len(contexts)
    print(f"\n[Test 2] Parallel inference ({num_samples} samples across {num_dies} dies)...")

    # Create dummy inputs for all samples
    dummy_inputs = [
        np.random.randn(1, 128, 3000).astype(np.float32)
        for _ in range(num_samples)
    ]

    die_names = list(contexts.keys())

    # Track results
    all_times = []
    lock = threading.Lock()

    def process_sample(sample_idx: int, mel_input: np.ndarray) -> float:
        """Process single sample on assigned die."""
        die_idx = sample_idx % num_dies
        die_name = die_names[die_idx]
        context = contexts[die_name]

        _, inference_time = run_encoder_on_die(context, mel_input)

        with lock:
            all_times.append((sample_idx, die_name, inference_time))

        return inference_time

    # Run parallel inference
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_dies) as executor:
        futures = [
            executor.submit(process_sample, i, dummy_inputs[i])
            for i in range(num_samples)
        ]

        for future in as_completed(futures):
            future.result()

    total_time = time.time() - start_time

    # Print per-sample results
    print("\n  Per-sample results:")
    all_times.sort(key=lambda x: x[0])
    for sample_idx, die_name, t in all_times:
        print(f"    Sample {sample_idx:2d} on {die_name}: {t*1000:.1f}ms")

    # Statistics
    throughput = num_samples / total_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Average per sample: {total_time/num_samples*1000:.1f}ms")

    return throughput, num_samples


def test_async_inference(
    contexts: Dict[str, DieContext],
    num_samples: int = 32
) -> Tuple[float, int]:
    """Test async inference with multiple in-flight requests per die."""
    import openvino as ov

    num_dies = len(contexts)
    nireq_per_die = 4  # Number of async requests per die

    print(f"\n[Test 3] Async inference ({num_samples} samples, {nireq_per_die} requests/die)...")

    # Create dummy inputs
    dummy_inputs = [
        np.random.randn(1, 128, 3000).astype(np.float32)
        for _ in range(num_samples)
    ]

    # Create async queues for each die
    async_queues = {}
    results_lock = threading.Lock()
    results = []

    def callback(request, userdata):
        """Callback when inference completes."""
        sample_idx, die_name, start_time = userdata
        inference_time = time.time() - start_time
        with results_lock:
            results.append((sample_idx, die_name, inference_time))

    for die_name, context in contexts.items():
        compiled = context.compiled_encoder
        try:
            queue = ov.AsyncInferQueue(compiled, nireq_per_die)
            queue.set_callback(callback)
            async_queues[die_name] = queue
            print(f"  ✓ Created AsyncInferQueue for {die_name} with {nireq_per_die} requests")
        except Exception as e:
            print(f"  ✗ Failed to create AsyncInferQueue for {die_name}: {e}")

    if not async_queues:
        print("  ✗ No async queues created")
        return 0, 0

    die_names = list(async_queues.keys())

    # Submit all samples
    start_time = time.time()

    for i, mel_input in enumerate(dummy_inputs):
        die_idx = i % len(die_names)
        die_name = die_names[die_idx]
        queue = async_queues[die_name]
        compiled = contexts[die_name].compiled_encoder

        sample_start = time.time()
        queue.start_async({compiled.inputs[0]: mel_input}, (i, die_name, sample_start))

    # Wait for all to complete
    for queue in async_queues.values():
        queue.wait_all()

    total_time = time.time() - start_time

    # Print results
    print(f"\n  Completed {len(results)}/{num_samples} samples")

    results.sort(key=lambda x: x[0])
    for sample_idx, die_name, t in results[:10]:  # First 10
        print(f"    Sample {sample_idx:2d} on {die_name}: {t*1000:.1f}ms")
    if len(results) > 10:
        print(f"    ... and {len(results) - 10} more")

    throughput = num_samples / total_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Average per sample: {total_time/num_samples*1000:.1f}ms")

    return throughput, num_samples


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-die distribution for Whisper encoder"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to OpenVINO Whisper model directory"
    )
    parser.add_argument(
        "--dies",
        type=str,
        default=None,
        help="Comma-separated list of dies to use (e.g., 'NPU.0,NPU.1')"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples for throughput tests"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Multi-Die Distribution Test")
    print("=" * 60)

    # Discover dies
    print("\n[1/5] Discovering NPU dies...")
    available_dies = discover_npu_dies()

    if not available_dies:
        print("  ✗ No NPU dies found!")
        print("  Run 01_check_npu_devices.py first")
        return 1

    print(f"  Found {len(available_dies)} dies: {available_dies}")

    # Select dies to use
    if args.dies:
        dies = [d.strip() for d in args.dies.split(",")]
        invalid = [d for d in dies if d not in available_dies]
        if invalid:
            print(f"  ✗ Invalid dies: {invalid}")
            return 1
    else:
        dies = available_dies

    print(f"  Using dies: {dies}")

    # Find encoder model
    model_path = Path(args.model_path)
    encoder_path = model_path / "encoder_model.xml"

    if not encoder_path.exists():
        encoder_path = model_path / "encoder.xml"
    if not encoder_path.exists():
        print(f"\n  ✗ Encoder not found at {model_path}")
        return 1

    print(f"\n[2/5] Found encoder: {encoder_path.name}")

    # Load encoder on all dies
    print("\n[3/5] Loading encoder on dies...")
    contexts = load_encoder_on_dies(encoder_path, dies)

    if not contexts:
        print("  ✗ Failed to load encoder on any die")
        return 1

    print(f"\n  ✓ Loaded on {len(contexts)} dies")

    # Run tests
    print("\n[4/5] Running inference tests...")

    # Test 1: Sequential
    seq_results = test_sequential_inference(contexts, num_samples=4)

    # Test 2: Parallel with threads
    par_throughput, par_samples = test_parallel_inference(contexts, args.num_samples)

    # Test 3: Async inference
    async_throughput, async_samples = test_async_inference(contexts, args.num_samples * 2)

    # Summary
    print("\n" + "=" * 60)
    print("[5/5] Summary")
    print("=" * 60)

    print(f"\n  Dies used: {len(contexts)}")
    for die_name, ctx in contexts.items():
        avg_time = ctx.total_time / ctx.infer_count if ctx.infer_count > 0 else 0
        print(f"    {die_name}: {ctx.infer_count} inferences, avg {avg_time*1000:.1f}ms")

    print(f"\n  Parallel throughput: {par_throughput:.1f} samples/sec")
    print(f"  Async throughput: {async_throughput:.1f} samples/sec")

    # Calculate scaling
    if len(seq_results) > 0:
        single_die_times = list(seq_results.values())[0]
        single_die_avg = np.mean(single_die_times[1:])  # Skip warmup
        single_die_throughput = 1.0 / single_die_avg

        print(f"\n  Single-die throughput: {single_die_throughput:.1f} samples/sec")
        print(f"  Multi-die speedup (parallel): {par_throughput / single_die_throughput:.2f}x")
        print(f"  Multi-die speedup (async): {async_throughput / single_die_throughput:.2f}x")
        print(f"  Ideal speedup: {len(contexts)}x")

    print("\n" + "=" * 60)
    print("Result: Multi-die encoder inference tested!")
    print("=" * 60)
    print("\nNext step: Run 05_full_whisper_benchmark.py for complete pipeline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
