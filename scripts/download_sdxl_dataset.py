#!/usr/bin/env python3
"""
Download and prepare COCO 2014 dataset for SDXL MLPerf benchmark.

This script:
1. Downloads COCO 2014 validation images and annotations
2. Extracts 5000 captions (one per image) as required by MLPerf
3. Generates latents for deterministic evaluation
4. Creates the expected directory structure

Requirements:
    pip install requests tqdm pycocotools torch diffusers

Usage:
    python scripts/download_sdxl_dataset.py --output data/coco2014
    python scripts/download_sdxl_dataset.py --output data/coco2014 --generate-latents
"""

import argparse
import json
import logging
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# COCO 2014 URLs
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# MLPerf uses 5000 samples
MLPERF_SAMPLE_COUNT = 5000

# Latent dimensions for SDXL
LATENT_CHANNELS = 4
LATENT_SIZE = 128  # 1024 // 8


def download_file(url: str, output_path: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("requests and tqdm required. Install: pip install requests tqdm")
        return False

    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {desc or url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def prepare_captions(
    annotations_file: Path,
    output_file: Path,
    num_samples: int = MLPERF_SAMPLE_COUNT,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Extract captions from COCO annotations.

    MLPerf requires exactly one caption per image for 5000 images.

    Args:
        annotations_file: Path to COCO captions JSON
        output_file: Output TSV file path
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility

    Returns:
        List of (image_id, caption) tuples
    """
    logger.info(f"Preparing captions from {annotations_file}")

    with open(annotations_file) as f:
        coco = json.load(f)

    # Group captions by image
    image_captions = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(ann['caption'])

    # Build image_id to filename mapping
    image_files = {img['id']: img['file_name'] for img in coco['images']}

    # Select images with captions
    valid_images = [img_id for img_id in image_captions if img_id in image_files]

    # Random sample for MLPerf
    random.seed(seed)
    if len(valid_images) > num_samples:
        selected_images = random.sample(valid_images, num_samples)
    else:
        selected_images = valid_images
        logger.warning(f"Only {len(valid_images)} images available, less than {num_samples}")

    # Select one caption per image (first one for consistency)
    samples = []
    for img_id in selected_images:
        caption = image_captions[img_id][0]  # Take first caption
        filename = image_files[img_id]
        samples.append((str(img_id), caption, filename))

    # Save as TSV
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("id\tcaption\tfilename\n")
        for img_id, caption, filename in samples:
            # Clean caption (remove tabs and newlines)
            caption = caption.replace('\t', ' ').replace('\n', ' ').strip()
            f.write(f"{img_id}\t{caption}\t{filename}\n")

    logger.info(f"Saved {len(samples)} captions to {output_file}")

    return [(s[0], s[1]) for s in samples]


def generate_latents(
    output_file: Path,
    num_samples: int = MLPERF_SAMPLE_COUNT,
    seed: int = 42,
) -> None:
    """
    Generate deterministic latents for MLPerf evaluation.

    Using fixed latents ensures reproducibility across different
    hardware and implementations.

    Args:
        output_file: Output file path (.pt or .npy)
        num_samples: Number of latents to generate
        seed: Random seed
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        logger.error("torch and numpy required. Install: pip install torch numpy")
        return

    logger.info(f"Generating {num_samples} latents...")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate latents
    latents = torch.randn(
        num_samples,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        dtype=torch.float16
    )

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix == '.npy':
        np.save(output_file, latents.numpy())
    else:
        torch.save(latents, output_file)

    logger.info(f"Saved latents to {output_file}")
    logger.info(f"Shape: {latents.shape}, dtype: {latents.dtype}")


def create_image_symlinks(
    coco_images_dir: Path,
    output_images_dir: Path,
    captions_file: Path,
) -> None:
    """Create symlinks to COCO images for the selected samples."""
    import pandas as pd

    logger.info("Creating image symlinks...")

    output_images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(captions_file, sep='\t')

    count = 0
    for _, row in df.iterrows():
        filename = row['filename']
        src = coco_images_dir / filename
        dst = output_images_dir / filename

        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
            count += 1

    logger.info(f"Created {count} symlinks")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare COCO 2014 dataset for SDXL MLPerf"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/coco2014",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=MLPERF_SAMPLE_COUNT,
        help="Number of samples to extract"
    )
    parser.add_argument(
        "--generate-latents",
        action="store_true",
        help="Generate deterministic latents"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading (use existing files)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    try:
        # Download COCO validation images
        if not args.skip_download:
            val_zip = downloads_dir / "val2014.zip"
            if not download_file(COCO_VAL_IMAGES_URL, val_zip, "COCO val2014 images"):
                sys.exit(1)

            # Extract images
            if not (output_dir / "val2014").exists():
                if not extract_zip(val_zip, output_dir):
                    sys.exit(1)

            # Download annotations
            ann_zip = downloads_dir / "annotations_trainval2014.zip"
            if not download_file(COCO_ANNOTATIONS_URL, ann_zip, "COCO annotations"):
                sys.exit(1)

            # Extract annotations
            if not (output_dir / "annotations").exists():
                if not extract_zip(ann_zip, output_dir):
                    sys.exit(1)

        # Prepare captions
        annotations_file = output_dir / "annotations" / "captions_val2014.json"
        if not annotations_file.exists():
            logger.error(f"Annotations file not found: {annotations_file}")
            sys.exit(1)

        captions_file = output_dir / "captions" / "captions.tsv"
        prepare_captions(
            annotations_file,
            captions_file,
            num_samples=args.num_samples,
            seed=args.seed,
        )

        # Create image symlinks
        try:
            import pandas
            create_image_symlinks(
                output_dir / "val2014",
                output_dir / "images",
                captions_file,
            )
        except ImportError:
            logger.warning("pandas not installed, skipping image symlinks")

        # Generate latents if requested
        if args.generate_latents:
            latents_file = output_dir / "latents" / "latents.pt"
            generate_latents(
                latents_file,
                num_samples=args.num_samples,
                seed=args.seed,
            )

        logger.info("\nDataset preparation complete!")
        logger.info(f"\nDirectory structure:")
        logger.info(f"  {output_dir}/")
        logger.info(f"    captions/captions.tsv  ({args.num_samples} prompts)")
        logger.info(f"    images/                (reference images)")
        if args.generate_latents:
            logger.info(f"    latents/latents.pt     (deterministic latents)")

        logger.info(f"\nUsage:")
        logger.info(f"  mlperf-ov run --model sdxl --dataset-path {output_dir} ...")

    except KeyboardInterrupt:
        logger.info("\nCancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
