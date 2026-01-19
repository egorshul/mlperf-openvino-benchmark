#!/usr/bin/env python3
"""
Download and convert Stable Diffusion XL model to OpenVINO IR format.

This script:
1. Downloads SDXL 1.0 Base from HuggingFace
2. Converts all components to OpenVINO IR using optimum-intel
3. Saves in the expected directory structure

Requirements:
    pip install optimum[openvino] diffusers transformers accelerate

Usage:
    python scripts/download_sdxl_model.py --output models/sdxl
    python scripts/download_sdxl_model.py --output models/sdxl --dtype fp16
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HuggingFace model ID
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import optimum
    except ImportError:
        missing.append("optimum[openvino]")

    try:
        import diffusers
    except ImportError:
        missing.append("diffusers")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import openvino
    except ImportError:
        missing.append("openvino")

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False

    return True


def download_and_convert_sdxl(
    output_dir: str,
    model_id: str = SDXL_MODEL_ID,
    dtype: str = "fp16",
    compress_weights: bool = False,
):
    """
    Download SDXL from HuggingFace and convert to OpenVINO IR.

    Args:
        output_dir: Output directory for OpenVINO models
        model_id: HuggingFace model ID
        dtype: Model precision (fp16 or fp32)
        compress_weights: Whether to apply INT8 weight compression
    """
    from optimum.intel import OVStableDiffusionXLPipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading and converting SDXL from {model_id}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Precision: {dtype}")

    # Export options
    export_kwargs = {}

    if compress_weights:
        from optimum.intel import OVWeightQuantizationConfig
        export_kwargs["quantization_config"] = OVWeightQuantizationConfig(
            bits=8,
            sym=False,
        )
        logger.info("Applying INT8 weight compression")

    # Download and convert
    logger.info("This may take several minutes...")

    try:
        # Load and export to OpenVINO
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            export=True,
            compile=False,  # Don't compile yet, just export
            **export_kwargs
        )

        # Save all components
        logger.info(f"Saving OpenVINO models to {output_path}")
        pipeline.save_pretrained(str(output_path))

        logger.info("Conversion complete!")

        # List saved files
        logger.info("\nSaved model components:")
        for component_dir in output_path.iterdir():
            if component_dir.is_dir():
                xml_files = list(component_dir.glob("*.xml"))
                if xml_files:
                    logger.info(f"  {component_dir.name}/")
                    for xml in xml_files:
                        bin_file = xml.with_suffix(".bin")
                        size_mb = bin_file.stat().st_size / (1024 * 1024) if bin_file.exists() else 0
                        logger.info(f"    {xml.name} ({size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


def download_tokenizers(output_dir: str):
    """Download and save tokenizers separately."""
    from transformers import CLIPTokenizer

    output_path = Path(output_dir)

    # Tokenizer 1 (CLIP)
    logger.info("Downloading CLIP tokenizer...")
    tokenizer_1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_1.save_pretrained(str(output_path / "tokenizer"))

    # Tokenizer 2 (OpenCLIP) - SDXL uses the same tokenizer class
    logger.info("Downloading OpenCLIP tokenizer...")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )
    tokenizer_2.save_pretrained(str(output_path / "tokenizer_2"))

    logger.info("Tokenizers saved")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert SDXL model to OpenVINO IR"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/sdxl",
        help="Output directory for OpenVINO models"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=SDXL_MODEL_ID,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model precision"
    )
    parser.add_argument(
        "--compress-weights",
        action="store_true",
        help="Apply INT8 weight compression"
    )
    parser.add_argument(
        "--tokenizers-only",
        action="store_true",
        help="Only download tokenizers (skip model conversion)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    try:
        if args.tokenizers_only:
            download_tokenizers(args.output)
        else:
            download_and_convert_sdxl(
                output_dir=args.output,
                model_id=args.model_id,
                dtype=args.dtype,
                compress_weights=args.compress_weights,
            )
            # Also save tokenizers
            download_tokenizers(args.output)

        logger.info("\nDone! Model saved to: " + args.output)
        logger.info("\nUsage:")
        logger.info(f"  mlperf-ov run --model sdxl --model-path {args.output} ...")

    except KeyboardInterrupt:
        logger.info("\nDownload cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
