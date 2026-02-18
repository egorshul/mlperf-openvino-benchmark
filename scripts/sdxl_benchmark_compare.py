#!/usr/bin/env python3
"""Benchmark: GenAI (text_enc+unet=NPU, vae=CPU) vs Optimum (all NPU).

Measures per-stage timing breakdown: text_encode, denoise (20 steps), vae_decode.
Runs 3 iterations each, reports median.

Usage:
    python scripts/sdxl_benchmark_compare.py [--device NPU.0] [--runs 3]
"""

import argparse
import sys
import time
from statistics import median

import numpy as np

# ── MLCommons parameters ──────────────────────────────────────────────────
MODEL_PATH = "./models/stable-diffusion-xl-base-1.0-openvino"
IMAGE_SIZE = 1024
GUIDANCE_SCALE = 8.0
NUM_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)
PROMPT = "A cinematic photo of a red panda wearing a wizard hat in a forest"
SEED = 42


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


# ── GenAI pipeline ────────────────────────────────────────────────────────

def bench_genai(device: str, num_runs: int):
    """GenAI Text2ImagePipeline: text_enc+unet on NPU, vae on CPU."""
    import openvino_genai

    print("[GenAI] Loading model ...", file=sys.stderr, flush=True)
    pipe = openvino_genai.Text2ImagePipeline(MODEL_PATH)
    pipe.reshape(1, IMAGE_SIZE, IMAGE_SIZE, GUIDANCE_SCALE)

    vae_device = "CPU" if "NPU" in device else device
    print(f"[GenAI] Compiling: text_enc={device}, unet={device}, vae={vae_device}",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    pipe.compile(device, device, vae_device)
    compile_time = time.perf_counter() - t0
    print(f"[GenAI] Compiled in {_fmt(compile_time)}", file=sys.stderr, flush=True)

    # Use callback to measure denoise boundaries.
    # text_encode = generate_start → first callback
    # denoise     = first callback → last callback end
    # vae_decode  = last callback end → generate_end
    step_times = []

    def callback(step, num_steps, _latent):
        step_times.append(time.perf_counter())
        return False

    results = []
    for i in range(num_runs):
        step_times.clear()
        t_start = time.perf_counter()
        pipe.generate(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_images_per_prompt=1,
            rng_seed=SEED,
            callback=callback,
        )
        t_end = time.perf_counter()

        # step_times[0] is called AFTER step 0 completes.
        # text_encode happens before step 0 starts,
        # but we can only measure: start → first_callback = text_encode + step_0
        # For a rough split: attribute first callback gap to text_encode + 1 step.
        first_cb = step_times[0] if step_times else t_end
        last_cb = step_times[-1] if step_times else t_start

        # Approximate: text_encode ≈ start → first_cb - avg_step_time
        avg_step = (last_cb - first_cb) / max(len(step_times) - 1, 1) if len(step_times) > 1 else 0
        text_enc = max(first_cb - t_start - avg_step, 0)
        denoise = last_cb - first_cb + avg_step  # all steps
        vae_dec = t_end - last_cb
        total = t_end - t_start

        tag = " (warmup)" if i == 0 else ""
        print(f"  run {i}{tag}: total={_fmt(total)}  "
              f"text_enc≈{_fmt(text_enc)}  denoise≈{_fmt(denoise)}  "
              f"vae={_fmt(vae_dec)}", file=sys.stderr, flush=True)
        results.append({
            "total": total, "text_enc": text_enc,
            "denoise": denoise, "vae_dec": vae_dec,
        })

    return compile_time, results


# ── Optimum pipeline ──────────────────────────────────────────────────────

def bench_optimum(device: str, num_runs: int):
    """OVStableDiffusionXLPipeline: all submodels on NPU."""
    import torch
    from optimum.intel import OVStableDiffusionXLPipeline
    from diffusers import EulerDiscreteScheduler

    print("[Optimum] Loading model ...", file=sys.stderr, flush=True)
    pipeline = OVStableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH, compile=False, load_in_8bit=False,
        ov_config={"EXECUTION_MODE_HINT": "ACCURACY"},
    )
    pipeline.reshape(
        batch_size=1, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_images_per_prompt=1,
    )

    print(f"[Optimum] Compiling on {device} ...", file=sys.stderr, flush=True)
    pipeline.to(device)
    t0 = time.perf_counter()
    pipeline.compile()
    compile_time = time.perf_counter() - t0
    print(f"[Optimum] Compiled in {_fmt(compile_time)}", file=sys.stderr, flush=True)

    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="leading", steps_offset=1,
        prediction_type="epsilon", use_karras_sigmas=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    if hasattr(pipeline, "watermark"):
        pipeline.watermark = None

    # Monkey-patch to measure per-stage timing
    _orig_text_enc = pipeline.text_encoder.__class__.__call__
    _orig_text_enc2 = pipeline.text_encoder_2.__class__.__call__
    _orig_unet = pipeline.unet.__class__.__call__
    _orig_vae = pipeline.vae_decoder.__class__.__call__

    timings = {"text_enc": [], "unet_calls": [], "vae_dec": []}

    def _wrap(orig, key):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            result = orig(*a, **kw)
            timings[key].append(time.perf_counter() - t)
            return result
        return wrapper

    pipeline.text_encoder.__class__.__call__ = _wrap(_orig_text_enc, "text_enc")
    pipeline.text_encoder_2.__class__.__call__ = _wrap(_orig_text_enc2, "text_enc")
    pipeline.unet.__class__.__call__ = _wrap(_orig_unet, "unet_calls")
    pipeline.vae_decoder.__class__.__call__ = _wrap(_orig_vae, "vae_dec")

    results = []
    for i in range(num_runs):
        for v in timings.values():
            v.clear()

        t_start = time.perf_counter()
        pipeline(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE, width=IMAGE_SIZE,
            output_type="np",
            generator=torch.Generator().manual_seed(SEED),
        )
        t_end = time.perf_counter()

        text_enc = sum(timings["text_enc"])
        denoise = sum(timings["unet_calls"])
        vae_dec = sum(timings["vae_dec"])
        total = t_end - t_start

        tag = " (warmup)" if i == 0 else ""
        print(f"  run {i}{tag}: total={_fmt(total)}  "
              f"text_enc={_fmt(text_enc)}  denoise={_fmt(denoise)}  "
              f"vae={_fmt(vae_dec)}", file=sys.stderr, flush=True)
        results.append({
            "total": total, "text_enc": text_enc,
            "denoise": denoise, "vae_dec": vae_dec,
        })

    # Restore original methods
    pipeline.text_encoder.__class__.__call__ = _orig_text_enc
    pipeline.text_encoder_2.__class__.__call__ = _orig_text_enc2
    pipeline.unet.__class__.__call__ = _orig_unet
    pipeline.vae_decoder.__class__.__call__ = _orig_vae

    return compile_time, results


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(genai_compile, genai_results, opt_compile, opt_results):
    # Skip warmup (run 0), take median of the rest
    def med(runs, key):
        vals = [r[key] for r in runs[1:]] if len(runs) > 1 else [r[key] for r in runs]
        return median(vals)

    print("\n" + "=" * 68)
    print(f"{'':30s} {'GenAI':>15s} {'Optimum':>15s}")
    print(f"{'':30s} {'NPU+NPU+CPU':>15s} {'all NPU':>15s}")
    print("-" * 68)
    print(f"{'Compile time':30s} {_fmt(genai_compile):>15s} {_fmt(opt_compile):>15s}")
    print("-" * 68)

    for key, label in [
        ("text_enc", "Text encode"),
        ("denoise", f"Denoise ({NUM_INFERENCE_STEPS} steps)"),
        ("vae_dec", "VAE decode"),
        ("total", "Total inference"),
    ]:
        g = med(genai_results, key)
        o = med(opt_results, key)
        diff = ((o - g) / g * 100) if g > 0 else 0
        sign = "+" if diff > 0 else ""
        print(f"{label:30s} {_fmt(g):>15s} {_fmt(o):>15s}  ({sign}{diff:.1f}%)")

    print("=" * 68)
    g_total = med(genai_results, "total")
    o_total = med(opt_results, "total")
    winner = "GenAI" if g_total < o_total else "Optimum"
    print(f"Winner: {winner} (by {_fmt(abs(g_total - o_total))})")


def main():
    parser = argparse.ArgumentParser(description="SDXL: GenAI vs Optimum benchmark")
    parser.add_argument("--device", default="NPU.0", help="Target device (default: NPU.0)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3, first is warmup)")
    args = parser.parse_args()

    print(f"Device: {args.device}, Runs: {args.runs} (first is warmup)\n",
          file=sys.stderr, flush=True)

    genai_compile, genai_results = bench_genai(args.device, args.runs)
    print(file=sys.stderr)
    opt_compile, opt_results = bench_optimum(args.device, args.runs)

    print_summary(genai_compile, genai_results, opt_compile, opt_results)


if __name__ == "__main__":
    main()
