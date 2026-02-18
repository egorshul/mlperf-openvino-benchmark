#!/usr/bin/env python3
"""Standalone SDXL inference on a single accelerator die.

All MLCommons parameters are hardcoded. Single argument: the prompt.

Usage:
    python scripts/sdxl_multi_die_infer.py "A photo of a cat on a windowsill"
"""

import sys
import time

import numpy as np
from PIL import Image

# ── MLCommons-mandated parameters ──────────────────────────────────────────
MODEL_PATH = "./models/stable-diffusion-xl-base-1.0-openvino"
DEVICE = "NPU.0"
IMAGE_SIZE = 1024
GUIDANCE_SCALE = 8.0
NUM_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)
SEED = 0
OUTPUT_FILE = "sdxl_output.png"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} \"<prompt>\"", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]

    # ── 1. Load & compile ──────────────────────────────────────────────────
    from optimum.intel import OVStableDiffusionXLPipeline

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr, flush=True)

    pipeline = OVStableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        compile=False,
        load_in_8bit=False,
        ov_config={"EXECUTION_MODE_HINT": "ACCURACY"},
    )

    # ── 2. Reshape for batch=1, 1024×1024 ─────────────────────────────────
    pipeline.reshape(
        batch_size=1,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_images_per_prompt=1,
    )

    # ── 3. Route to die & compile ──────────────────────────────────────────
    print(f"Compiling on {DEVICE} ...", file=sys.stderr, flush=True)
    pipeline.to(DEVICE)
    pipeline.compile()

    # ── 4. Scheduler (MLCommons reference) ─────────────────────────────────
    from diffusers import EulerDiscreteScheduler

    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="leading",
        steps_offset=1,
        prediction_type="epsilon",
        use_karras_sigmas=False,
    )

    pipeline.set_progress_bar_config(disable=True)
    if hasattr(pipeline, "watermark"):
        pipeline.watermark = None

    # ── 5. Inference ───────────────────────────────────────────────────────
    import torch

    print(f"Generating: \"{prompt}\" ...", file=sys.stderr, flush=True)
    t0 = time.time()

    result = pipeline(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        output_type="np",
        generator=torch.Generator().manual_seed(SEED),
    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    # ── 6. Save ────────────────────────────────────────────────────────────
    image = result.images[0]
    if isinstance(image, np.ndarray) and image.max() <= 1.0:
        image = (image * 255).round().astype(np.uint8)

    Image.fromarray(image).save(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
