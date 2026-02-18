#!/usr/bin/env python3
"""Standalone SDXL inference using OpenVINO GenAI Text2ImagePipeline.

All MLCommons parameters are hardcoded. Single argument: the prompt.

Usage:
    python scripts/sdxl_genai_infer.py "A photo of a cat on a windowsill"
"""

import sys
import time

from PIL import Image

# ── MLCommons-mandated parameters ──────────────────────────────────────────
MODEL_PATH = "./models/stable-diffusion-xl-base-1.0-openvino"
DEVICE = "NPU.0"
IMAGE_SIZE = 1024
NUM_IMAGES = 1
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

    # ── 1. Create pipeline (no device → no auto-compile) ──────────────────
    import openvino_genai

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr, flush=True)

    pipe = openvino_genai.Text2ImagePipeline(MODEL_PATH)

    # ── 2. Reshape to static shapes (required for NPU) ────────────────────
    #   reshape(num_images_per_prompt, height, width, guidance_scale)
    print(f"Reshaping to {NUM_IMAGES}×{IMAGE_SIZE}×{IMAGE_SIZE}, "
          f"guidance_scale={GUIDANCE_SCALE} ...", file=sys.stderr, flush=True)

    pipe.reshape(NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, GUIDANCE_SCALE)

    # ── 3. Compile on target device ────────────────────────────────────────
    print(f"Compiling on {DEVICE} ...", file=sys.stderr, flush=True)

    pipe.compile(DEVICE)

    # ── 4. Progress callback ───────────────────────────────────────────────
    def callback(step, num_steps, latent):
        print(f"\r  Step {step + 1}/{num_steps}", end="", file=sys.stderr, flush=True)
        return False  # don't interrupt

    # ── 5. Inference ───────────────────────────────────────────────────────
    print(f"Generating: \"{prompt}\" ...", file=sys.stderr, flush=True)
    t0 = time.time()

    image_tensor = pipe.generate(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=NUM_IMAGES,
        rng_seed=SEED,
        callback=callback,
    )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s", file=sys.stderr)

    # ── 6. Save ────────────────────────────────────────────────────────────
    image = Image.fromarray(image_tensor.data[0])
    image.save(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
