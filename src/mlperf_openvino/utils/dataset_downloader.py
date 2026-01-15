"""
Dataset downloader utility for MLPerf OpenVINO Benchmark.

Downloads and prepares datasets required for MLPerf benchmarks:
- ImageNet 2012 validation set (50,000 images) for ResNet50
- LibriSpeech dev-clean + dev-other for Whisper

Based on official MLCommons scripts and documentation.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError

logger = logging.getLogger(__name__)


# Dataset registry with download URLs
DATASET_REGISTRY: Dict[str, Dict] = {
    "imagenet": {
        "description": "ImageNet 2012 validation set (50,000 images) for ResNet50",
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "filename": "ILSVRC2012_img_val.tar",
        "size_gb": 6.3,
        "num_samples": 50000,
        "val_map_url": "https://raw.githubusercontent.com/mlcommons/inference/master/vision/classification_and_detection/tools/val_map.txt",
    },
    "squad": {
        "description": "SQuAD v1.1 dataset for BERT Question Answering",
        "dev": {
            "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
            "filename": "dev-v1.1.json",
            "size_mb": 4.9,
            "num_samples": 10833,
        },
        "train": {
            "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
            "filename": "train-v1.1.json",
            "size_mb": 30,
            "num_samples": 87599,
        },
        "vocab": {
            "url": "https://zenodo.org/record/3733868/files/vocab.txt",
            "filename": "vocab.txt",
            "size_mb": 0.2,
        },
    },
    "openimages": {
        "description": "OpenImages validation set for RetinaNet Object Detection",
        "annotations": {
            # V5 validation annotations work for V6 as well
            "url": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            "filename": "validation-annotations-bbox.csv",
            "size_mb": 24,
        },
        "class_descriptions": {
            "url": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
            "filename": "class-descriptions-boxable.csv",
            "size_mb": 0.02,
        },
        "num_samples": 24781,  # Official MLPerf count from 365 class filtering
        "note": "Images downloaded from AWS S3 open-images-dataset bucket",
    },
    "librispeech": {
        "description": "LibriSpeech ASR corpus for Whisper (dev-clean + dev-other)",
        "dev-clean": {
            "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "filename": "dev-clean.tar.gz",
            "extracted_dir": "LibriSpeech/dev-clean",
            "size_mb": 337,
            "num_samples": 2703,
            "md5": "42e2234ba48799c1f50f24a7926300a1",
        },
        "dev-other": {
            "url": "https://www.openslr.org/resources/12/dev-other.tar.gz",
            "filename": "dev-other.tar.gz",
            "extracted_dir": "LibriSpeech/dev-other",
            "size_mb": 314,
            "num_samples": 2864,
            "md5": "c8d0bcc9cca99d4f8b62fcc847357931",
        },
        "test-clean": {
            "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "filename": "test-clean.tar.gz",
            "extracted_dir": "LibriSpeech/test-clean",
            "size_mb": 346,
            "num_samples": 2620,
            "md5": "32fa31d27d2e1c7c6744fb7a529f1ab0",
        },
        "test-other": {
            "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
            "filename": "test-other.tar.gz",
            "extracted_dir": "LibriSpeech/test-other",
            "size_mb": 328,
            "num_samples": 2939,
            "md5": "d09c181bba5cf717b3dee7d4d592af11",
        },
        # MLPerf Whisper uses dev-clean + dev-other
        "mlperf_subsets": ["dev-clean", "dev-other"],
        "mlperf_num_samples": 5567,
    },
    "whisper-mlperf": {
        "description": "Pre-processed LibriSpeech for MLPerf Whisper benchmark",
        "r2_url": "https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri",
        "download_cmd": (
            'bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/'
            'refs/heads/main/mlc-r2-downloader.sh) -d {output_dir} '
            'https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri'
        ),
    },
}


def _download_file(
    url: str,
    destination: str,
    show_progress: bool = True,
    expected_size_mb: Optional[float] = None
) -> None:
    """Download a file from URL with progress indication."""
    logger.info(f"Downloading from {url}")
    
    try:
        if show_progress:
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded / total_size) * 100)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\rDownloading: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", 
                          end="", flush=True)
                elif expected_size_mb:
                    downloaded_mb = downloaded / (1024 * 1024)
                    print(f"\rDownloading: {downloaded_mb:.1f}/{expected_size_mb:.0f} MB", 
                          end="", flush=True)
            
            urllib.request.urlretrieve(url, destination, progress_hook)
            print()
        else:
            urllib.request.urlretrieve(url, destination)
        
        logger.info(f"Downloaded to {destination}")
        
    except URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def _verify_md5(file_path: str, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    logger.info("Verifying checksum...")
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    
    if actual_md5 != expected_md5:
        logger.warning(f"Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False
    
    logger.info("Checksum verified")
    return True


def _extract_archive(archive_path: str, output_dir: str) -> str:
    """Extract tar.gz or tar archive."""
    logger.info(f"Extracting {archive_path}...")
    
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    
    if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
    elif str(archive_path).endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar:
            tar.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    logger.info(f"Extracted to {output_dir}")
    return str(output_dir)


def download_imagenet(
    output_dir: str,
    force: bool = False,
    preprocess: bool = True
) -> Dict[str, str]:
    """
    Download and prepare ImageNet 2012 validation dataset for ResNet50.
    
    Downloads from: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    
    Args:
        output_dir: Directory to save dataset
        force: Force re-download even if exists
        preprocess: Whether to preprocess images for ResNet50
        
    Returns:
        Dictionary with dataset paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_path / "imagenet"
    val_dir = data_dir / "val"
    val_map_file = data_dir / "val_map.txt"
    
    # Check if already downloaded and extracted
    if val_dir.exists() and val_map_file.exists() and not force:
        # Count images
        image_count = len(list(val_dir.glob("*.JPEG")))
        if image_count >= 50000:
            logger.info(f"ImageNet already exists at {data_dir} with {image_count} images")
            return {
                "data_path": str(data_dir),
                "val_dir": str(val_dir),
                "val_map": str(val_map_file),
                "num_samples": image_count,
            }
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ImageNet validation set
    dataset_info = DATASET_REGISTRY["imagenet"]
    archive_path = output_path / dataset_info["filename"]
    
    if not archive_path.exists() or force:
        logger.info(f"Downloading ImageNet validation set (~{dataset_info['size_gb']} GB)...")
        logger.info("This may take a while...")
        _download_file(
            dataset_info["url"],
            str(archive_path),
            expected_size_mb=dataset_info["size_gb"] * 1024
        )
    
    # Extract images to val directory
    if not list(val_dir.glob("*.JPEG")):
        logger.info("Extracting images...")
        _extract_archive(str(archive_path), str(val_dir))
    
    # Download val_map.txt (labels file)
    if not val_map_file.exists():
        logger.info("Downloading val_map.txt...")
        _download_file(dataset_info["val_map_url"], str(val_map_file))
    
    # Preprocess if requested
    if preprocess:
        preprocessed_dir = data_dir / "preprocessed"
        _preprocess_imagenet(val_dir, preprocessed_dir, val_map_file)
    
    image_count = len(list(val_dir.glob("*.JPEG")))
    logger.info(f"ImageNet dataset ready: {image_count} images")
    
    return {
        "data_path": str(data_dir),
        "val_dir": str(val_dir),
        "val_map": str(val_map_file),
        "num_samples": image_count,
    }


def _preprocess_imagenet(
    val_dir: Path,
    output_dir: Path,
    val_map_file: Path
) -> None:
    """
    Preprocess ImageNet images for ResNet50.
    
    Applies:
    - Resize to 256x256
    - Center crop to 224x224
    - Normalize with ImageNet mean/std
    - Save as numpy arrays
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("PIL/numpy not available, skipping preprocessing")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ImageNet preprocessing parameters
    resize_size = 256
    crop_size = 224
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)  # RGB
    
    # Read val_map.txt
    with open(val_map_file, 'r') as f:
        lines = f.readlines()
    
    logger.info(f"Preprocessing {len(lines)} images...")
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        img_name = parts[0]
        img_path = val_dir / img_name
        
        if not img_path.exists():
            continue
        
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        
        # Resize maintaining aspect ratio
        w, h = img.size
        if w < h:
            new_w = resize_size
            new_h = int(h * resize_size / w)
        else:
            new_h = resize_size
            new_w = int(w * resize_size / h)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        
        # Convert to numpy and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array - mean
        
        # Save preprocessed image
        output_file = output_dir / f"{img_name.replace('.JPEG', '.npy')}"
        np.save(output_file, img_array)
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Preprocessed {i + 1}/{len(lines)} images")
    
    logger.info(f"Preprocessing complete: {output_dir}")


def download_librispeech(
    output_dir: str,
    subset: str = "mlperf",
    force: bool = False
) -> Dict[str, str]:
    """
    Download LibriSpeech dataset for Whisper benchmark.
    
    For MLPerf submissions, use subset="mlperf" which downloads
    both dev-clean and dev-other.
    
    Args:
        output_dir: Directory to save dataset
        subset: "mlperf" (dev-clean + dev-other), "dev-clean", "dev-other", etc.
        force: Force re-download
        
    Returns:
        Dictionary with dataset paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle MLPerf subset (dev-clean + dev-other)
    if subset == "mlperf":
        logger.info("Downloading MLPerf LibriSpeech (dev-clean + dev-other)...")
        
        results = {"subsets": []}
        total_samples = 0
        
        for sub in DATASET_REGISTRY["librispeech"]["mlperf_subsets"]:
            sub_result = _download_librispeech_subset(output_path, sub, force)
            results["subsets"].append(sub_result)
            total_samples += sub_result.get("num_samples", 0)
        
        # Create combined directory with symlinks
        mlperf_dir = output_path / "librispeech" / "mlperf"
        mlperf_dir.mkdir(parents=True, exist_ok=True)
        
        results["data_path"] = str(output_path / "librispeech")
        results["num_samples"] = total_samples
        results["note"] = "MLPerf Whisper uses dev-clean + dev-other combined"
        
        logger.info(f"LibriSpeech MLPerf ready: {total_samples} samples")
        return results
    
    # Single subset
    return _download_librispeech_subset(output_path, subset, force)


def _download_librispeech_subset(
    output_path: Path,
    subset: str,
    force: bool = False
) -> Dict[str, str]:
    """Download a single LibriSpeech subset."""
    dataset_info = DATASET_REGISTRY["librispeech"].get(subset)
    if not dataset_info or not isinstance(dataset_info, dict):
        available = [k for k, v in DATASET_REGISTRY["librispeech"].items() 
                     if isinstance(v, dict) and "url" in v]
        raise ValueError(f"Unknown subset: {subset}. Available: {available}")
    
    data_dir = output_path / "librispeech" / subset
    audio_dir = data_dir / "audio"
    transcript_file = data_dir / "transcripts.txt"
    
    # Check if already exists
    if data_dir.exists() and transcript_file.exists() and not force:
        logger.info(f"LibriSpeech {subset} already exists at {data_dir}")
        return {
            "data_path": str(data_dir),
            "audio_dir": str(audio_dir),
            "transcript_path": str(transcript_file),
            "num_samples": dataset_info["num_samples"],
        }
    
    # Download
    archive_path = output_path / dataset_info["filename"]
    
    if not archive_path.exists() or force:
        logger.info(f"Downloading LibriSpeech {subset} (~{dataset_info['size_mb']} MB)...")
        _download_file(
            dataset_info["url"],
            str(archive_path),
            expected_size_mb=dataset_info["size_mb"]
        )
        
        # Verify checksum
        if dataset_info.get("md5"):
            if not _verify_md5(str(archive_path), dataset_info["md5"]):
                archive_path.unlink()
                raise RuntimeError("Checksum verification failed")
    
    # Extract
    _extract_archive(str(archive_path), str(output_path))
    
    # Organize: move from LibriSpeech/dev-clean to librispeech/dev-clean
    extracted_dir = output_path / dataset_info["extracted_dir"]
    
    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    # Process and create transcripts
    _process_librispeech(extracted_dir, audio_dir, transcript_file)
    
    logger.info(f"LibriSpeech {subset} ready at {data_dir}")
    
    return {
        "data_path": str(data_dir),
        "audio_dir": str(audio_dir),
        "transcript_path": str(transcript_file),
        "num_samples": dataset_info["num_samples"],
    }


def _process_librispeech(
    source_dir: Path,
    audio_dir: Path,
    transcript_file: Path
) -> None:
    """
    Process LibriSpeech directory structure.
    
    LibriSpeech format:
        speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        speaker_id/chapter_id/speaker_id-chapter_id.trans.txt
    
    Creates:
        audio/speaker_id-chapter_id-utterance_id.flac (copies or symlinks)
        transcripts.txt
    """
    logger.info("Processing LibriSpeech files...")
    
    transcripts = []
    audio_count = 0
    
    for speaker_dir in sorted(source_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            # Read transcripts
            chapter_transcripts = {}
            for trans_file in chapter_dir.glob("*.trans.txt"):
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                chapter_transcripts[parts[0]] = parts[1]
            
            # Process audio files
            for audio_file in sorted(chapter_dir.glob("*.flac")):
                utterance_id = audio_file.stem
                
                # Copy audio file
                dest_audio = audio_dir / audio_file.name
                if not dest_audio.exists():
                    shutil.copy2(audio_file, dest_audio)
                
                transcript = chapter_transcripts.get(utterance_id, "")
                transcripts.append((utterance_id, transcript))
                audio_count += 1
    
    # Write transcripts
    with open(transcript_file, 'w', encoding='utf-8') as f:
        for audio_id, transcript in transcripts:
            f.write(f"{audio_id} {transcript}\n")
    
    logger.info(f"Processed {audio_count} audio files")


def download_whisper_mlperf(output_dir: str) -> Dict[str, str]:
    """
    Download pre-processed LibriSpeech for MLPerf Whisper using official MLCommons script.
    
    Uses the R2 downloader from MLCommons.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        Dictionary with paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = DATASET_REGISTRY["whisper-mlperf"]["download_cmd"].format(
        output_dir=output_path / "whisper-dataset"
    )
    
    logger.info("Downloading MLPerf Whisper dataset using R2 downloader...")
    logger.info(f"Command: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return {
            "data_path": str(output_path / "whisper-dataset"),
            "note": "Downloaded using MLCommons R2 downloader",
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download: {e}")
        logger.info("Falling back to manual LibriSpeech download...")
        return download_librispeech(output_dir, subset="mlperf")


def download_squad(
    output_dir: str,
    subset: str = "dev",
    force: bool = False
) -> Dict[str, str]:
    """
    Download SQuAD v1.1 dataset for BERT Question Answering.

    Args:
        output_dir: Directory to save dataset
        subset: "dev" or "train"
        force: Force re-download

    Returns:
        Dictionary with dataset paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "squad"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASET_REGISTRY["squad"]

    # Download dev set by default
    if subset in dataset_info:
        subset_info = dataset_info[subset]
        data_file = data_dir / subset_info["filename"]

        if not data_file.exists() or force:
            logger.info(f"Downloading SQuAD {subset} set...")
            _download_file(
                subset_info["url"],
                str(data_file),
                expected_size_mb=subset_info.get("size_mb")
            )

    # Also download vocab file
    if "vocab" in dataset_info:
        vocab_info = dataset_info["vocab"]
        vocab_file = data_dir / vocab_info["filename"]

        if not vocab_file.exists() or force:
            logger.info("Downloading BERT vocab file...")
            _download_file(vocab_info["url"], str(vocab_file))

    logger.info(f"SQuAD dataset ready at {data_dir}")

    return {
        "data_path": str(data_dir),
        "dev_file": str(data_dir / "dev-v1.1.json"),
        "vocab_file": str(data_dir / "vocab.txt"),
        "num_samples": dataset_info["dev"]["num_samples"],
    }


def _download_openimages_image(args) -> Optional[str]:
    """Download a single OpenImages image from S3."""
    image_id, images_dir = args

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        # Fallback to HTTP download
        url = f"https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg"
        dest = images_dir / f"{image_id}.jpg"
        try:
            urllib.request.urlretrieve(url, str(dest))
            return image_id
        except Exception:
            return None

    try:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        dest = images_dir / f"{image_id}.jpg"
        s3.download_file('open-images-dataset', f'validation/{image_id}.jpg', str(dest))
        return image_id
    except Exception:
        return None


def download_openimages(
    output_dir: str,
    force: bool = False,
    max_images: Optional[int] = None,
    num_workers: int = 8
) -> Dict[str, str]:
    """
    Download OpenImages validation set for RetinaNet Object Detection.

    Uses official MLCommons approach:
    1. Download annotations from Google Cloud Storage
    2. Extract image IDs from annotations
    3. Download images from AWS S3 open-images-dataset bucket
    4. Convert annotations to COCO format

    Based on: https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection

    Args:
        output_dir: Directory to save dataset
        force: Force re-download
        max_images: Maximum number of images to download (None = all validation images)
        num_workers: Number of parallel download workers

    Returns:
        Dictionary with dataset paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "openimages"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "validation" / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(images_dir.glob("*.jpg"))
    coco_annotations = annotations_dir / "openimages-mlperf.json"
    # Only use quick check if max_images is explicitly specified
    # Otherwise, we need to filter by MLPerf classes to determine actual count
    if max_images and len(existing_images) >= max_images and coco_annotations.exists() and not force:
        logger.info(f"OpenImages already downloaded: {len(existing_images)} images")
        return {
            "data_path": str(data_dir),
            "annotations_file": str(coco_annotations),
            "images_dir": str(images_dir),
            "num_samples": len(existing_images),
        }

    # URLs for OpenImages v5 annotations (compatible with MLPerf)
    ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
    CLASS_NAMES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

    # Step 1: Download annotations
    annotations_file = annotations_dir / "validation-annotations-bbox.csv"
    if not annotations_file.exists() or force:
        logger.info("Downloading annotations...")
        _download_file(ANNOTATIONS_URL, str(annotations_file))

    class_names_file = annotations_dir / "class-descriptions-boxable.csv"
    if not class_names_file.exists() or force:
        logger.info("Downloading class descriptions...")
        _download_file(CLASS_NAMES_URL, str(class_names_file))

    # Step 2: Get MLPerf classes and filter annotations
    # Official MLPerf uses these 365 classes from openimages_mlperf.sh
    MLPERF_CLASSES = [
        "Airplane", "Antelope", "Apple", "Backpack", "Balloon", "Banana",
        "Barrel", "Baseball bat", "Baseball glove", "Bee", "Beer", "Bench",
        "Bicycle", "Bicycle helmet", "Bicycle wheel", "Billboard", "Book",
        "Bookcase", "Boot", "Bottle", "Bowl", "Bowling equipment", "Box",
        "Boy", "Brassiere", "Bread", "Broccoli", "Bronze sculpture",
        "Bull", "Bus", "Bust", "Butterfly", "Cabinetry", "Cake",
        "Camel", "Camera", "Candle", "Candy", "Cannon", "Canoe",
        "Carrot", "Cart", "Castle", "Cat", "Cattle", "Cello", "Chair",
        "Cheese", "Chest of drawers", "Chicken", "Christmas tree", "Coat",
        "Cocktail", "Coffee", "Coffee cup", "Coffee table", "Coin",
        "Common sunflower", "Computer keyboard", "Computer monitor",
        "Convenience store", "Cookie", "Countertop", "Cowboy hat", "Crab",
        "Crocodile", "Cucumber", "Cupboard", "Curtain", "Deer", "Desk",
        "Dinosaur", "Dog", "Doll", "Dolphin", "Door", "Dragonfly",
        "Drawer", "Dress", "Drum", "Duck", "Eagle", "Earrings",
        "Egg (Food)", "Elephant", "Falcon", "Fedora", "Flag", "Flowerpot",
        "Football", "Football helmet", "Fork", "Fountain", "French fries",
        "French horn", "Frog", "Giraffe", "Girl", "Glasses", "Goat",
        "Goggles", "Goldfish", "Gondola", "Goose", "Grape", "Grapefruit",
        "Guitar", "Hamburger", "Handbag", "Harbor seal", "Headphones",
        "Helicopter", "High heels", "Hiking equipment", "Horse", "House",
        "Houseplant", "Human arm", "Human beard", "Human body", "Human ear",
        "Human eye", "Human face", "Human foot", "Human hair", "Human hand",
        "Human head", "Human leg", "Human mouth", "Human nose", "Ice cream",
        "Jacket", "Jeans", "Jellyfish", "Juice", "Kitchen & dining room table",
        "Kite", "Lamp", "Lantern", "Laptop", "Lavender (Plant)", "Lemon",
        "Light bulb", "Lighthouse", "Lily", "Lion", "Lipstick", "Lizard",
        "Man", "Maple", "Microphone", "Mirror", "Mixing bowl", "Mobile phone",
        "Monkey", "Motorcycle", "Muffin", "Mug", "Mule", "Mushroom",
        "Musical keyboard", "Necklace", "Nightstand", "Office building",
        "Orange", "Owl", "Oyster", "Paddle", "Palm tree", "Parachute",
        "Parrot", "Pen", "Penguin", "Personal flotation device", "Piano",
        "Picture frame", "Pig", "Pillow", "Pizza", "Plate", "Platter",
        "Porch", "Poster", "Pumpkin", "Rabbit", "Rifle", "Roller skates",
        "Rose", "Salad", "Sandal", "Saucer", "Saxophone", "Scarf",
        "Sea lion", "Sea turtle", "Sheep", "Shelf", "Shirt", "Shorts",
        "Shrimp", "Sink", "Skateboard", "Ski", "Skull", "Skyscraper",
        "Snake", "Sock", "Sofa bed", "Sparrow", "Spider", "Spoon",
        "Sports uniform", "Squirrel", "Stairs", "Stool", "Strawberry",
        "Street light", "Studio couch", "Suit", "Sun hat", "Sunglasses",
        "Surfboard", "Sushi", "Swan", "Swimming pool", "Swimwear", "Tank",
        "Tap", "Taxi", "Tea", "Teddy bear", "Television", "Tent", "Tie",
        "Tiger", "Tin can", "Tire", "Toilet", "Tomato", "Tortoise",
        "Tower", "Traffic light", "Train", "Tripod", "Truck", "Trumpet",
        "Umbrella", "Van", "Vase", "Vehicle registration plate", "Violin",
        "Wall clock", "Waste container", "Watch", "Whale", "Wheel",
        "Wheelchair", "Whiteboard", "Window", "Wine", "Wine glass",
        "Woman", "Zebra", "Zucchini",
    ]

    import csv

    # Load class name to LabelName mapping
    logger.info("Loading class descriptions...")
    class_name_to_label = {}
    with open(class_names_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_name, display_name = row[0], row[1]
                class_name_to_label[display_name] = label_name

    # Get LabelNames for MLPerf classes
    mlperf_labels = set()
    for class_name in MLPERF_CLASSES:
        if class_name in class_name_to_label:
            mlperf_labels.add(class_name_to_label[class_name])

    logger.info(f"Found {len(mlperf_labels)} MLPerf class labels")

    # Extract image IDs that have MLPerf class annotations
    logger.info("Extracting image IDs with MLPerf classes...")
    image_ids = []
    seen_ids = set()

    with open(annotations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row['LabelName']
            if label_name in mlperf_labels:
                image_id = row['ImageID']
                if image_id not in seen_ids:
                    seen_ids.add(image_id)
                    image_ids.append(image_id)

    logger.info(f"Found {len(image_ids)} images with MLPerf classes")

    # Apply max_images limit only if explicitly specified
    if max_images and max_images < len(image_ids):
        image_ids = image_ids[:max_images]
        logger.info(f"Limited to {max_images} images (user specified)")

    # Check if already complete (all filtered images downloaded)
    existing = set(p.stem for p in images_dir.glob("*.jpg"))
    if len(existing) >= len(image_ids) and coco_annotations.exists() and not force:
        logger.info(f"OpenImages already complete: {len(existing)} images (MLPerf filtered: {len(image_ids)})")
        return {
            "data_path": str(data_dir),
            "annotations_file": str(coco_annotations),
            "images_dir": str(images_dir),
            "num_samples": len(existing),
        }

    # Step 3: Download images from S3
    to_download = [img_id for img_id in image_ids if img_id not in existing]

    if to_download:
        logger.info(f"Downloading {len(to_download)} images ({len(existing)} already exist)...")
        _download_openimages_from_s3(to_download, images_dir, num_workers)

    # Step 4: Convert to COCO format
    logger.info("Converting annotations to COCO format...")
    _convert_openimages_to_coco(
        annotations_file=annotations_file,
        class_names_file=class_names_file,
        image_ids=image_ids,
        images_dir=images_dir,
        output_file=coco_annotations,
    )

    final_count = len(list(images_dir.glob("*.jpg")))
    logger.info(f"OpenImages ready: {final_count} images")

    return {
        "data_path": str(data_dir),
        "annotations_file": str(coco_annotations),
        "images_dir": str(images_dir),
        "num_samples": final_count,
    }


def _download_openimages_from_s3(
    image_ids: List[str],
    images_dir: Path,
    num_workers: int = 8
) -> None:
    """Download images from AWS S3 open-images-dataset bucket."""
    import time
    import ssl
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Disable SSL verification globally for this download
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Try to use requests for better handling
    requests_session = None
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        requests_session = requests.Session()
        requests_session.verify = False  # Disable SSL verification

        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Configure retries
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        requests_session.mount('https://', adapter)
        requests_session.mount('http://', adapter)
    except ImportError:
        pass

    def download_one(image_id: str) -> Optional[str]:
        dest = images_dir / f"{image_id}.jpg"
        if dest.exists():
            return image_id

        url = f"https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg"

        for attempt in range(3):
            try:
                if requests_session is not None:
                    response = requests_session.get(url, timeout=60)
                    response.raise_for_status()
                    with open(dest, 'wb') as f:
                        f.write(response.content)
                else:
                    # Fallback with SSL disabled
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, context=ssl_context, timeout=60) as resp:
                        with open(dest, 'wb') as f:
                            f.write(resp.read())

                return image_id

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    if dest.exists():
                        dest.unlink()
                    return None

        return None

    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_one, img_id): img_id for img_id in image_ids}

        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded += 1
            else:
                failed += 1

            total = downloaded + failed
            if total % 100 == 0 or total == len(image_ids):
                print(f"\rDownloaded: {downloaded}/{len(image_ids)}, failed: {failed}", end="", flush=True)

    print()
    logger.info(f"Downloaded {downloaded} images, {failed} failed")


def _convert_openimages_to_coco(
    annotations_file: Path,
    class_names_file: Path,
    image_ids: List[str],
    images_dir: Path,
    output_file: Path,
) -> None:
    """Convert OpenImages annotations to COCO format."""
    import csv
    import json

    # Load class names
    class_map = {}  # LabelName -> class_id
    class_names = {}  # class_id -> display_name
    with open(class_names_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 2:
                label_name, display_name = row[0], row[1]
                class_map[label_name] = i + 1  # 1-indexed
                class_names[i + 1] = display_name

    # Load annotations
    image_id_set = set(image_ids)
    annotations_by_image = {}

    with open(annotations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageID']
            if image_id not in image_id_set:
                continue

            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []

            annotations_by_image[image_id].append({
                'LabelName': row['LabelName'],
                'XMin': float(row['XMin']),
                'YMin': float(row['YMin']),
                'XMax': float(row['XMax']),
                'YMax': float(row['YMax']),
                'IsOccluded': int(row.get('IsOccluded', 0)),
                'IsTruncated': int(row.get('IsTruncated', 0)),
                'IsGroupOf': int(row.get('IsGroupOf', 0)),
            })

    # Build COCO format
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': cid, 'name': name} for cid, name in class_names.items()],
    }

    annotation_id = 1

    for img_idx, image_id in enumerate(image_ids):
        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            # Default dimensions if can't read
            width, height = 800, 600

        coco['images'].append({
            'id': img_idx + 1,
            'file_name': f"{image_id}.jpg",
            'width': width,
            'height': height,
        })

        # Add annotations for this image
        for ann in annotations_by_image.get(image_id, []):
            label_name = ann['LabelName']
            if label_name not in class_map:
                continue

            # Convert normalized coords to pixels
            x_min = ann['XMin'] * width
            y_min = ann['YMin'] * height
            x_max = ann['XMax'] * width
            y_max = ann['YMax'] * height

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            coco['annotations'].append({
                'id': annotation_id,
                'image_id': img_idx + 1,
                'category_id': class_map[label_name],
                'bbox': [x_min, y_min, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': ann['IsGroupOf'],
            })
            annotation_id += 1

    # Write COCO JSON
    with open(output_file, 'w') as f:
        json.dump(coco, f)

    logger.info(f"COCO annotations: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def download_dataset(
    dataset_name: str,
    output_dir: str,
    subset: Optional[str] = None,
    force: bool = False
) -> Dict[str, str]:
    """
    Download a dataset by name.

    Args:
        dataset_name: "imagenet", "squad", "openimages", or "librispeech"
        output_dir: Directory to save dataset
        subset: Optional subset name
        force: Force re-download

    Returns:
        Dictionary with dataset paths
    """
    if dataset_name == "imagenet":
        return download_imagenet(output_dir, force)
    elif dataset_name == "squad":
        return download_squad(output_dir, subset or "dev", force)
    elif dataset_name == "openimages":
        return download_openimages(output_dir, force)
    elif dataset_name == "librispeech":
        return download_librispeech(output_dir, subset or "mlperf", force)
    elif dataset_name == "whisper-mlperf":
        return download_whisper_mlperf(output_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def list_available_datasets() -> Dict[str, str]:
    """List available datasets."""
    return {
        "imagenet": DATASET_REGISTRY["imagenet"]["description"],
        "squad": DATASET_REGISTRY["squad"]["description"],
        "openimages": DATASET_REGISTRY["openimages"]["description"],
        "librispeech": DATASET_REGISTRY["librispeech"]["description"],
    }


def get_dataset_info(dataset_name: str) -> Dict:
    """Get dataset information."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name]


def redownload_corrupted_openimages(data_path: str, num_workers: int = 8) -> int:
    """
    Re-download corrupted OpenImages files.

    Checks for corrupted images (< 1KB or unreadable) and re-downloads them.

    Args:
        data_path: Path to OpenImages dataset directory
        num_workers: Number of parallel download workers

    Returns:
        Number of files re-downloaded
    """
    from PIL import Image

    data_dir = Path(data_path)
    images_dir = data_dir / "validation" / "data"
    if not images_dir.exists():
        images_dir = data_dir / "images"

    if not images_dir.exists():
        logger.error(f"Images directory not found in {data_path}")
        return 0

    # Find corrupted images
    corrupted = []
    logger.info("Checking for corrupted images...")

    for img_path in images_dir.glob("*.jpg"):
        try:
            # Check file size
            if img_path.stat().st_size < 1000:
                corrupted.append(img_path.stem)
                continue

            # Try to open image
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted.append(img_path.stem)

    if not corrupted:
        logger.info("No corrupted images found")
        return 0

    logger.info(f"Found {len(corrupted)} corrupted images, re-downloading...")

    # Delete corrupted files
    for image_id in corrupted:
        img_path = images_dir / f"{image_id}.jpg"
        if img_path.exists():
            img_path.unlink()

    # Re-download
    _download_openimages_from_s3(corrupted, images_dir, num_workers)

    # Also clear preprocessed cache for these images
    cache_dir = data_dir / "preprocessed_cache"
    if cache_dir.exists():
        for image_id in corrupted:
            cache_file = cache_dir / f"{image_id}.npy"
            if cache_file.exists():
                cache_file.unlink()
        logger.info(f"Cleared {len(corrupted)} cached preprocessed files")

    return len(corrupted)
