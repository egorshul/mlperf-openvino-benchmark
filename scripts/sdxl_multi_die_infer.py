#!/usr/bin/env python3
"""Standalone SDXL multi-die inference script.

Replicates the full SDXLMultiDieSUT pipeline: device discovery,
per-die compilation with reshape, scheduler setup, and parallel
inference across multiple accelerator dies.

Usage:
    python scripts/sdxl_multi_die_infer.py \
        --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
        --device NPU \
        --prompt "A photo of a cat sitting on a windowsill at sunset"

    # Multiple prompts (one image per prompt):
    python scripts/sdxl_multi_die_infer.py \
        --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
        --device NPU.0,NPU.1 \
        --prompt "A red sports car" "A blue ocean wave" "A mountain at dawn"

    # Single die, custom steps:
    python scripts/sdxl_multi_die_infer.py \
        --model-path ./models/stable-diffusion-xl-base-1.0-openvino \
        --device NPU.0 \
        --steps 30 --guidance-scale 7.5 \
        --prompt "An astronaut riding a horse on Mars"
"""

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def discover_device_dies(device: str) -> List[str]:
    """Return sorted list of sub-device identifiers (e.g. ['NPU.0', 'NPU.1'])."""
    import openvino as ov

    core = ov.Core()
    pattern = re.compile(rf"^{re.escape(device)}\.(\d+)$")
    dies = sorted(d for d in core.available_devices if pattern.match(d))
    return dies


def resolve_device_dies(device: str) -> List[str]:
    """Parse --device value into a list of concrete die identifiers."""
    if device.upper() == "CPU":
        return ["CPU"]

    # Comma-separated: "NPU.0,NPU.2"
    if "," in device:
        return [p.strip() for p in device.split(",")]

    # Single specific die: "NPU.0"
    if re.match(r"^.+\.\d+$", device):
        return [device]

    # Bare accelerator name: "NPU" → auto-discover all dies
    dies = discover_device_dies(device)
    if not dies:
        print(f"[WARN] No {device} dies found, falling back to bare device", file=sys.stderr)
        return [device]
    return dies


# ---------------------------------------------------------------------------
# Pipeline loading (per die)
# ---------------------------------------------------------------------------

def load_pipeline_for_device(
    model_path: str,
    die: str,
    batch_size: int,
    image_size: int,
) -> Any:
    """Load and compile OVStableDiffusionXLPipeline for a single die."""
    from optimum.intel import OVStableDiffusionXLPipeline

    is_cpu = die.upper() == "CPU"
    ov_config = {"EXECUTION_MODE_HINT": "ACCURACY"}

    print(f"  Loading model for {die} ...", file=sys.stderr, flush=True)

    if is_cpu and batch_size <= 1:
        # CPU with batch=1: compile immediately
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_path, compile=True, load_in_8bit=False,
            ov_config=ov_config,
        )
    else:
        # Accelerator or batch>1: reshape → route to die → compile
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_path, compile=False, load_in_8bit=False,
            ov_config=ov_config,
        )
        pipeline.reshape(
            batch_size=batch_size,
            height=image_size,
            width=image_size,
            num_images_per_prompt=1,
        )
        if not is_cpu:
            pipeline.to(die)
        pipeline.compile()

    # --- Scheduler: EulerDiscrete matching MLPerf reference ---
    try:
        from diffusers import EulerDiscreteScheduler

        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="leading",
            steps_offset=1,
            prediction_type="epsilon",
            use_karras_sigmas=False,
        )
    except Exception as exc:
        print(f"  [WARN] EulerDiscreteScheduler setup failed on {die}: {exc}",
              file=sys.stderr)

    # Disable progress bar and watermark
    pipeline.set_progress_bar_config(disable=True)
    if hasattr(pipeline, "watermark"):
        pipeline.watermark = None

    print(f"  {die} ready", file=sys.stderr, flush=True)
    return pipeline


# ---------------------------------------------------------------------------
# Single-prompt inference
# ---------------------------------------------------------------------------

def generate_image(
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    image_size: int,
    seed: int,
) -> np.ndarray:
    """Run a single inference and return uint8 HWC numpy image."""
    import torch

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "height": image_size,
        "width": image_size,
        "output_type": "np",
        "generator": torch.Generator().manual_seed(seed),
    }

    image = pipeline(**pipe_kwargs).images[0]

    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).round().astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
    else:
        image = np.array(image)

    return image


# ---------------------------------------------------------------------------
# Multi-die parallel inference
# ---------------------------------------------------------------------------

def run_multi_die(
    model_path: str,
    device: str,
    prompts: List[str],
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    image_size: int,
    batch_size: int,
    seed: int,
    output_dir: str,
) -> None:
    # --- 1. Resolve dies ---
    device_dies = resolve_device_dies(device)
    num_dies = len(device_dies)
    print(f"[SDXL] Device(s): {', '.join(device_dies)}", file=sys.stderr)

    # --- 2. Load one pipeline per die ---
    pipelines: List[Tuple[str, Any, threading.Lock]] = []
    for die in device_dies:
        pipeline = load_pipeline_for_device(model_path, die, batch_size, image_size)
        pipelines.append((die, pipeline, threading.Lock()))

    print(f"[SDXL] {num_dies} pipeline(s) compiled", file=sys.stderr)

    # --- 3. Distribute prompts round-robin across dies ---
    total = len(prompts)
    results: List[Tuple[int, np.ndarray]] = [None] * total  # type: ignore[list-item]

    start_time = time.time()

    if num_dies == 1:
        # Sequential path (single die)
        _name, pipeline, _lock = pipelines[0]
        for i, prompt in enumerate(prompts):
            img = generate_image(
                pipeline, prompt, negative_prompt,
                guidance_scale, num_inference_steps, image_size,
                seed=seed + i,
            )
            results[i] = (i, img)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\r  [{i + 1}/{total}] {rate:.2f} img/s",
                  end="", file=sys.stderr, flush=True)
    else:
        # Parallel path
        die_batches: List[List[Tuple[int, str]]] = [[] for _ in range(num_dies)]
        for i, prompt in enumerate(prompts):
            die_batches[i % num_dies].append((i, prompt))

        completed = 0
        completed_lock = threading.Lock()

        def _worker(die_idx: int, batch: List[Tuple[int, str]]):
            nonlocal completed
            _name, pipeline, die_lock = pipelines[die_idx]
            for idx, prompt in batch:
                with die_lock:
                    img = generate_image(
                        pipeline, prompt, negative_prompt,
                        guidance_scale, num_inference_steps, image_size,
                        seed=seed + idx,
                    )
                results[idx] = (idx, img)
                with completed_lock:
                    completed += 1

        with ThreadPoolExecutor(max_workers=num_dies) as pool:
            futures = [
                pool.submit(_worker, idx, batch)
                for idx, batch in enumerate(die_batches)
                if batch
            ]
            # Progress monitor
            while not all(f.done() for f in futures):
                with completed_lock:
                    c = completed
                elapsed = time.time() - start_time
                rate = c / elapsed if elapsed > 0 else 0
                print(f"\r  [{c}/{total}] {rate:.2f} img/s",
                      end="", file=sys.stderr, flush=True)
                time.sleep(0.5)
            # Propagate exceptions
            for f in futures:
                f.result()

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\r  [{total}/{total}] {rate:.2f} img/s — done in {elapsed:.1f}s",
          file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # --- 4. Save images ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, img_array in results:
        fname = out_path / f"sdxl_{idx:04d}.png"
        Image.fromarray(img_array).save(fname)
        print(f"  Saved {fname}")

    print(f"\n{total} image(s) saved to {out_path}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone SDXL multi-die inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to exported OpenVINO SDXL model directory",
    )
    parser.add_argument(
        "--device", default="CPU",
        help=(
            "Device specification: CPU, NPU (all dies), NPU.0 (one die), "
            "NPU.0,NPU.2 (selected dies)"
        ),
    )
    parser.add_argument(
        "--prompt", nargs="+", required=True,
        help="One or more prompts (each produces one image)",
    )
    parser.add_argument(
        "--negative-prompt",
        default="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
        help="Negative prompt",
    )
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output_sdxl")

    args = parser.parse_args()

    run_multi_die(
        model_path=args.model_path,
        device=args.device,
        prompts=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
