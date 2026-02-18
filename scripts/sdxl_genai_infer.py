#!/usr/bin/env python3
"""Standalone SDXL inference using OpenVINO GenAI Text2ImagePipeline.

All MLCommons parameters are hardcoded. Single argument: the prompt.

Usage:
    python scripts/sdxl_genai_infer.py "A photo of a cat on a windowsill"
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
OUTPUT_FILE = "sdxl_output_genai.png"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} \"<prompt>\"", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]

    # ── 1. Create pipeline ─────────────────────────────────────────────────
    import openvino_genai

    print(f"Loading model from {MODEL_PATH} on {DEVICE} ...",
          file=sys.stderr, flush=True)

    pipe = openvino_genai.Text2ImagePipeline(MODEL_PATH, DEVICE)

    # ── 2. Progress callback ───────────────────────────────────────────────
    def callback(step, num_steps, latent):
        print(f"\r  Step {step + 1}/{num_steps}", end="", file=sys.stderr, flush=True)
        return False  # don't interrupt

    # ── 3. Inference ───────────────────────────────────────────────────────
    print(f"Generating: \"{prompt}\" ...", file=sys.stderr, flush=True)
    t0 = time.time()

    image_tensor = pipe.generate(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        rng_seed=SEED,
        callback=callback,
    )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s", file=sys.stderr)

    # ── 4. Save ────────────────────────────────────────────────────────────
    image = Image.fromarray(image_tensor.data[0])
    image.save(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
