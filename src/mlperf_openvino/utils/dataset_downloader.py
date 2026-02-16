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
        "num_samples": 24781,  # Official MLPerf count from 264 class filtering
        "note": "Images downloaded from AWS S3 open-images-dataset bucket",
    },
    "librispeech": {
        "description": "LibriSpeech ASR corpus for Whisper (dev-clean + dev-other)",
        "dev-clean": {
            "url": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "filename": "dev-clean.tar.gz",
            "extracted_dir": "LibriSpeech/dev-clean",
            "size_mb": 337,
            "num_samples": 2703,
            "md5": "42e2234ba48799c1f50f24a7926300a1",
        },
        "dev-other": {
            "url": "http://www.openslr.org/resources/12/dev-other.tar.gz",
            "filename": "dev-other.tar.gz",
            "extracted_dir": "LibriSpeech/dev-other",
            "size_mb": 314,
            "num_samples": 2864,
            "md5": "c8d0bcc9cca99d4f8b62fcc847357931",
        },
        "test-clean": {
            "url": "http://www.openslr.org/resources/12/test-clean.tar.gz",
            "filename": "test-clean.tar.gz",
            "extracted_dir": "LibriSpeech/test-clean",
            "size_mb": 346,
            "num_samples": 2620,
            "md5": "32fa31d27d2e1c7c6744fb7a529f1ab0",
        },
        "test-other": {
            "url": "http://www.openslr.org/resources/12/test-other.tar.gz",
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
    "coco2014": {
        "description": "COCO 2014 captions dataset for SDXL text-to-image generation",
        "captions": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "filename": "annotations_trainval2014.zip",
            "size_mb": 241,
        },
        "images": {
            "url": "http://images.cocodataset.org/zips/val2014.zip",
            "filename": "val2014.zip",
            "size_gb": 6.0,
        },
        # MLCommons pre-computed files for official submission
        "fid_statistics": {
            "url": "https://github.com/mlcommons/inference/raw/master/text_to_image/tools/val2014.npz",
            "filename": "val2014.npz",
            "size_mb": 10,
        },
        "latents": {
            # Single shared latent tensor (1, 4, 128, 128) ~256KB
            "url": "https://github.com/mlcommons/inference/raw/master/text_to_image/tools/latents.pt",
            "filename": "latents.pt",
            "size_mb": 1,
        },
        "captions_tsv": {
            "url": "https://github.com/mlcommons/inference/raw/master/text_to_image/coco2014/captions/captions_source.tsv",
            "filename": "captions_source.tsv",
            "size_mb": 1,
        },
        "num_samples": 5000,  # MLPerf uses 5000 samples
        "note": "For MLPerf SDXL benchmark (closed division)",
    },
    "cnn-dailymail": {
        "description": "CNN-DailyMail v3.0.0 for MLPerf LLM benchmark (Llama 3.1 8B summarization)",
        "huggingface": {
            "dataset_id": "cnn_dailymail",
            "version": "3.0.0",
        },
        "num_samples_datacenter": 13368,
        "num_samples_edge": 5000,
        "eval_file": "cnn_eval.json",
        "note": "MLCommons Inference v5.1 — text summarization task for Llama 3.1 8B",
    },
    "coco2017": {
        "description": "COCO 2017 validation set for SSD-ResNet34 Object Detection",
        "images": {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "filename": "val2017.zip",
            "size_gb": 1.0,
        },
        "annotations": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "filename": "annotations_trainval2017.zip",
            "size_mb": 252,
        },
        "num_samples": 5000,  # COCO 2017 val set
        "note": "For MLPerf SSD-ResNet34 benchmark",
    },
}


def _download_file(
    url: str,
    destination: str,
    show_progress: bool = True,
    expected_size_mb: Optional[float] = None,
    min_size_bytes: int = 0,
) -> None:
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

        # Validate download: check file exists and meets minimum size
        dest_path = Path(destination)
        if not dest_path.exists():
            raise RuntimeError(f"Download completed but file not found: {destination}")
        actual_size = dest_path.stat().st_size
        if min_size_bytes > 0 and actual_size < min_size_bytes:
            dest_path.unlink()
            raise RuntimeError(
                f"Downloaded file is too small ({actual_size} bytes, expected >= {min_size_bytes}). "
                f"The server may have returned an error page instead of the actual file."
            )

        logger.info(f"Downloaded to {destination} ({actual_size / (1024*1024):.1f} MB)")

    except URLError as e:
        # Clean up partial download
        dest_path = Path(destination)
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")


def _verify_md5(file_path: str, expected_md5: str) -> bool:
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "imagenet"
    val_dir = data_dir / "val"
    val_map_file = data_dir / "val_map.txt"

    if val_dir.exists() and val_map_file.exists() and not force:
        image_count = len(list(val_dir.glob("*.JPEG")))
        if image_count >= 50000:
            logger.info(f"ImageNet already exists at {data_dir} with {image_count} images")
            return {
                "data_path": str(data_dir),
                "val_dir": str(val_dir),
                "val_map": str(val_map_file),
                "num_samples": image_count,
            }

    data_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

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

    if not list(val_dir.glob("*.JPEG")):
        logger.info("Extracting images...")
        _extract_archive(str(archive_path), str(val_dir))

    if not val_map_file.exists():
        logger.info("Downloading val_map.txt...")
        _download_file(dataset_info["val_map_url"], str(val_map_file))

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
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("PIL/numpy not available, skipping preprocessing")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    resize_size = 256
    crop_size = 224
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)  # RGB

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

        img = Image.open(img_path).convert('RGB')

        w, h = img.size
        if w < h:
            new_w = resize_size
            new_h = int(h * resize_size / w)
        else:
            new_h = resize_size
            new_w = int(w * resize_size / h)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))

        img_array = np.array(img, dtype=np.float32)
        img_array = img_array - mean

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
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if subset == "mlperf":
        logger.info("Downloading MLPerf LibriSpeech (dev-clean + dev-other)...")

        results = {"subsets": []}
        total_samples = 0
        all_manifest_entries = []

        for sub in DATASET_REGISTRY["librispeech"]["mlperf_subsets"]:
            sub_result = _download_librispeech_subset(output_path, sub, force)
            results["subsets"].append(sub_result)
            total_samples += sub_result.get("num_samples", 0)

            manifest_path = sub_result.get("manifest_path")
            if manifest_path and Path(manifest_path).exists():
                with open(manifest_path, 'r') as f:
                    entries = json.load(f)
                    all_manifest_entries.extend(entries)

        mlperf_dir = output_path / "librispeech" / "mlperf"
        mlperf_dir.mkdir(parents=True, exist_ok=True)

        combined_manifest = mlperf_dir / "manifest.json"
        with open(combined_manifest, 'w') as f:
            json.dump(all_manifest_entries, f, indent=2)
        logger.info(f"Created combined manifest with {len(all_manifest_entries)} entries")

        combined_transcripts = mlperf_dir / "transcripts.txt"
        with open(combined_transcripts, 'w', encoding='utf-8') as f:
            for entry in all_manifest_entries:
                utterance_id = entry.get("utterance_id", "")
                text = entry.get("text", "")
                f.write(f"{utterance_id} {text}\n")

        results["data_path"] = str(mlperf_dir)
        results["manifest_path"] = str(combined_manifest)
        results["transcript_path"] = str(combined_transcripts)
        results["num_samples"] = total_samples
        results["note"] = "MLPerf Whisper uses dev-clean + dev-other combined"

        logger.info(f"LibriSpeech MLPerf ready: {total_samples} samples at {mlperf_dir}")
        return results

    return _download_librispeech_subset(output_path, subset, force)


def _download_librispeech_subset(
    output_path: Path,
    subset: str,
    force: bool = False
) -> Dict[str, str]:
    dataset_info = DATASET_REGISTRY["librispeech"].get(subset)
    if not dataset_info or not isinstance(dataset_info, dict):
        available = [k for k, v in DATASET_REGISTRY["librispeech"].items()
                     if isinstance(v, dict) and "url" in v]
        raise ValueError(f"Unknown subset: {subset}. Available: {available}")

    data_dir = output_path / "librispeech" / subset
    audio_dir = data_dir / "audio"
    transcript_file = data_dir / "transcripts.txt"
    manifest_file = data_dir / "manifest.json"

    if data_dir.exists() and transcript_file.exists() and not force:
        logger.info(f"LibriSpeech {subset} already exists at {data_dir}")
        actual_count = len(list(audio_dir.glob("*.flac"))) if audio_dir.exists() else 0
        return {
            "data_path": str(data_dir),
            "audio_dir": str(audio_dir),
            "transcript_path": str(transcript_file),
            "manifest_path": str(manifest_file) if manifest_file.exists() else None,
            "num_samples": actual_count or dataset_info["num_samples"],
        }

    archive_path = output_path / dataset_info["filename"]

    if not archive_path.exists() or force:
        logger.info(f"Downloading LibriSpeech {subset} (~{dataset_info['size_mb']} MB)...")
        _download_file(
            dataset_info["url"],
            str(archive_path),
            expected_size_mb=dataset_info["size_mb"]
        )

        if dataset_info.get("md5"):
            if not _verify_md5(str(archive_path), dataset_info["md5"]):
                archive_path.unlink()
                raise RuntimeError("Checksum verification failed")

    _extract_archive(str(archive_path), str(output_path))

    # Reorganize from LibriSpeech/dev-clean to librispeech/dev-clean
    extracted_dir = output_path / dataset_info["extracted_dir"]

    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    actual_count = _process_librispeech(extracted_dir, audio_dir, transcript_file, manifest_file)

    logger.info(f"LibriSpeech {subset} ready at {data_dir}")

    return {
        "data_path": str(data_dir),
        "audio_dir": str(audio_dir),
        "transcript_path": str(transcript_file),
        "manifest_path": str(manifest_file),
        "num_samples": actual_count,
    }


def _process_librispeech(
    source_dir: Path,
    audio_dir: Path,
    transcript_file: Path,
    manifest_file: Optional[Path] = None
) -> int:
    # LibriSpeech directory format:
    #   speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
    #   speaker_id/chapter_id/speaker_id-chapter_id.trans.txt
    import json

    try:
        import soundfile as sf
        has_soundfile = True
    except ImportError:
        has_soundfile = False

    logger.info("Processing LibriSpeech files...")

    transcripts = []
    manifest_entries = []
    audio_count = 0

    for speaker_dir in sorted(source_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_transcripts = {}
            for trans_file in chapter_dir.glob("*.trans.txt"):
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                chapter_transcripts[parts[0]] = parts[1]

            for audio_file in sorted(chapter_dir.glob("*.flac")):
                utterance_id = audio_file.stem

                dest_audio = audio_dir / audio_file.name
                if not dest_audio.exists():
                    shutil.copy2(audio_file, dest_audio)

                transcript = chapter_transcripts.get(utterance_id, "")
                transcripts.append((utterance_id, transcript))

                duration = 0.0
                if has_soundfile and dest_audio.exists():
                    try:
                        info = sf.info(str(dest_audio))
                        duration = info.duration
                    except Exception:
                        pass

                manifest_entries.append({
                    "audio_filepath": str(dest_audio),
                    "text": transcript,
                    "duration": duration,
                    "utterance_id": utterance_id,
                })

                audio_count += 1

    with open(transcript_file, 'w', encoding='utf-8') as f:
        for audio_id, transcript in transcripts:
            f.write(f"{audio_id} {transcript}\n")

    if manifest_file:
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_entries, f, indent=2)
        logger.info(f"Created manifest with {len(manifest_entries)} entries")

    logger.info(f"Processed {audio_count} audio files")
    return audio_count


def download_whisper_mlperf(output_dir: str) -> Dict[str, str]:
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "squad"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASET_REGISTRY["squad"]

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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "openimages"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "validation" / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list(images_dir.glob("*.jpg"))
    coco_annotations = annotations_dir / "openimages-mlperf.json"
    # Only use quick check if max_images is explicitly specified;
    # otherwise we need to filter by MLPerf classes to determine actual count
    if max_images and len(existing_images) >= max_images and coco_annotations.exists() and not force:
        logger.info(f"OpenImages already downloaded: {len(existing_images)} images")
        return {
            "data_path": str(data_dir),
            "annotations_file": str(coco_annotations),
            "images_dir": str(images_dir),
            "num_samples": len(existing_images),
        }

    ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
    CLASS_NAMES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

    annotations_file = annotations_dir / "validation-annotations-bbox.csv"
    if not annotations_file.exists() or force:
        logger.info("Downloading annotations...")
        _download_file(ANNOTATIONS_URL, str(annotations_file))

    class_names_file = annotations_dir / "class-descriptions-boxable.csv"
    if not class_names_file.exists() or force:
        logger.info("Downloading class descriptions...")
        _download_file(CLASS_NAMES_URL, str(class_names_file))

    # Official MLPerf 264 classes
    # See: https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/openimages_mlperf.sh
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

    logger.info("Loading class descriptions...")
    class_name_to_label = {}
    with open(class_names_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_name, display_name = row[0], row[1]
                class_name_to_label[display_name] = label_name

    mlperf_labels = set()
    missing_classes = []
    for class_name in MLPERF_CLASSES:
        if class_name in class_name_to_label:
            mlperf_labels.add(class_name_to_label[class_name])
        else:
            missing_classes.append(class_name)

    logger.info(f"Found {len(mlperf_labels)} MLPerf class labels in OpenImages class descriptions")
    if missing_classes:
        logger.info(f"Note: {len(missing_classes)} MLPerf classes not found in OpenImages boxable classes.")
        logger.info(f"  This is expected - not all model classes appear in the validation set.")
        logger.debug(f"  Missing classes: {missing_classes[:20]}{'...' if len(missing_classes) > 20 else ''}")

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

    if max_images and max_images < len(image_ids):
        image_ids = image_ids[:max_images]
        logger.info(f"Limited to {max_images} images (user specified)")

    existing = set(p.stem for p in images_dir.glob("*.jpg"))
    if len(existing) >= len(image_ids) and coco_annotations.exists() and not force:
        logger.info(f"OpenImages already complete: {len(existing)} images (MLPerf filtered: {len(image_ids)})")
        return {
            "data_path": str(data_dir),
            "annotations_file": str(coco_annotations),
            "images_dir": str(images_dir),
            "num_samples": len(existing),
        }

    to_download = [img_id for img_id in image_ids if img_id not in existing]

    if to_download:
        logger.info(f"Downloading {len(to_download)} images ({len(existing)} already exist)...")
        _download_openimages_from_s3(to_download, images_dir, num_workers)

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
) -> List[str]:
    import time
    import ssl
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Disable SSL verification for this download
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    requests_session = None
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        requests_session = requests.Session()
        requests_session.verify = False

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        requests_session.mount('https://', adapter)
        requests_session.mount('http://', adapter)
    except ImportError:
        pass

    DOWNLOAD_URLS = [
        "https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg",
        "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg",
    ]

    def download_one(image_id: str, force_redownload: bool = False) -> Optional[str]:
        dest = images_dir / f"{image_id}.jpg"

        if dest.exists() and not force_redownload:
            if dest.stat().st_size > 1000:
                return image_id
            else:
                dest.unlink()

        last_error = None
        for url_template in DOWNLOAD_URLS:
            url = url_template.format(image_id=image_id)

            for attempt in range(3):
                try:
                    if requests_session is not None:
                        response = requests_session.get(url, timeout=60)
                        response.raise_for_status()
                        content = response.content
                    else:
                        # Fallback with SSL disabled
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as resp:
                            content = resp.read()

                    if len(content) < 1000:
                        last_error = f"Content too small ({len(content)} bytes)"
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            break

                    with open(dest, 'wb') as f:
                        f.write(content)

                    if dest.stat().st_size > 1000:
                        return image_id
                    else:
                        dest.unlink()
                        last_error = "Written file too small"
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        break

                except Exception as e:
                    last_error = str(e)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        break

        if dest.exists():
            dest.unlink()
        return None

    downloaded = 0
    failed = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_one, img_id): img_id for img_id in image_ids}

        for future in as_completed(futures):
            img_id = futures[future]
            result = future.result()
            if result:
                downloaded += 1
            else:
                failed += 1
                failed_ids.append(img_id)

            total = downloaded + failed
            if total % 100 == 0 or total == len(image_ids):
                print(f"\rDownloaded: {downloaded}/{len(image_ids)}, failed: {failed}", end="", flush=True)

    print()
    logger.info(f"Downloaded {downloaded} images, {failed} failed")

    if failed_ids:
        failed_log = images_dir.parent.parent / "failed_downloads.txt"
        with open(failed_log, 'w') as f:
            for img_id in failed_ids:
                f.write(f"{img_id}\n")
        logger.warning(f"Failed image IDs saved to: {failed_log}")
        logger.warning(f"Failed images: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")

    return failed_ids


def _convert_openimages_to_coco(
    annotations_file: Path,
    class_names_file: Path,
    image_ids: List[str],
    images_dir: Path,
    output_file: Path,
) -> None:
    """Uses only the 264 MLPerf classes with sequential category_ids (1-264).
    This matches the MLPerf RetinaNet model output format where class indices
    correspond to the alphabetically sorted MLPerf class list."""
    import csv
    import json

    # MLPerf 264 classes, alphabetically sorted (this is the model's class order)
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

    display_to_label = {}
    with open(class_names_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_name, display_name = row[0], row[1]
                display_to_label[display_name] = label_name

    # Sequential category_ids (1-264) matching model output class indices (0-263) + 1
    class_map = {}
    class_names = {}

    for idx, display_name in enumerate(MLPERF_CLASSES):
        if display_name in display_to_label:
            label_name = display_to_label[display_name]
            class_id = idx + 1  # 1-indexed
            class_map[label_name] = class_id
            class_names[class_id] = display_name

    logger.info(f"Built class mapping for {len(class_map)} MLPerf classes (category_ids 1-{len(class_map)})")

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

    coco = {
        'info': {
            'description': 'OpenImages MLPerf subset',
            'version': '1.0',
            'year': 2024,
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': cid, 'name': name} for cid, name in class_names.items()],
    }

    annotation_id = 1

    for img_idx, image_id in enumerate(image_ids):
        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

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

    with open(output_file, 'w') as f:
        json.dump(coco, f)

    logger.info(f"COCO annotations: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def _resize_coco_images(
    source_dir: Path,
    output_dir: Path,
    prompts_file: Path,
    target_size: int = 1024
) -> int:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not installed, cannot resize images")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = []
    with open(prompts_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                image_ids.append(parts[0])

    logger.info(f"Resizing {len(image_ids)} images to {target_size}x{target_size}...")

    resized_count = 0
    for i, image_id in enumerate(image_ids):
        source_path = source_dir / f"COCO_val2014_{int(image_id):012d}.jpg"
        if not source_path.exists():
            source_path = source_dir / f"{image_id}.jpg"
            if not source_path.exists():
                continue

        output_path = output_dir / f"{image_id}.png"
        if output_path.exists():
            resized_count += 1
            continue

        try:
            with Image.open(source_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size
                min_dim = min(width, height)

                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))

                img = img.resize((target_size, target_size), Image.LANCZOS)

                img.save(output_path, 'PNG')
                resized_count += 1

        except Exception as e:
            logger.warning(f"Failed to resize {source_path}: {e}")

        if (i + 1) % 500 == 0:
            logger.info(f"Resized {i + 1}/{len(image_ids)} images")

    logger.info(f"Resized {resized_count} images to {output_dir}")
    return resized_count


def download_coco2014(
    output_dir: str,
    force: bool = False,
    download_images: bool = False,
    num_samples: int = 5000
) -> Dict[str, str]:
    import json
    import zipfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "coco2014"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASET_REGISTRY["coco2014"]

    annotations_zip = output_path / dataset_info["captions"]["filename"]
    annotations_dir = data_dir / "annotations"

    captions_file = annotations_dir / "captions_val2014.json"

    if not captions_file.exists() or force:
        if not annotations_zip.exists() or force:
            logger.info(f"Downloading COCO 2014 annotations (~{dataset_info['captions']['size_mb']} MB)...")
            _download_file(
                dataset_info["captions"]["url"],
                str(annotations_zip),
                expected_size_mb=dataset_info["captions"]["size_mb"]
            )

        logger.info("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    prompts_file = data_dir / "coco-1024.tsv"
    prompts_txt = data_dir / "prompts.txt"

    if not prompts_file.exists() or force:
        logger.info("Processing captions...")

        with open(captions_file, 'r') as f:
            coco_data = json.load(f)

        # Get one caption per image (first caption)
        image_to_caption = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_to_caption:
                image_to_caption[image_id] = ann['caption']

        id_to_filename = {}
        for img in coco_data['images']:
            id_to_filename[img['id']] = img['file_name']

        # TSV format: image_id<tab>caption
        with open(prompts_file, 'w', encoding='utf-8') as tsv_f, \
             open(prompts_txt, 'w', encoding='utf-8') as txt_f:

            count = 0
            for image_id, caption in image_to_caption.items():
                if count >= num_samples:
                    break

                caption = caption.strip().replace('\t', ' ').replace('\n', ' ')

                tsv_f.write(f"{image_id}\t{caption}\n")
                txt_f.write(f"{caption}\n")
                count += 1

        logger.info(f"Created prompts file with {count} samples")

    images_dir = data_dir / "coco-1024"

    if download_images:
        images_zip = output_path / dataset_info["images"]["filename"]
        val_dir = data_dir / "val2014"

        if not val_dir.exists() and not images_dir.exists():
            if not images_zip.exists() or force:
                logger.info(f"Downloading COCO 2014 validation images (~{dataset_info['images']['size_gb']} GB)...")
                logger.info("This may take a while...")
                _download_file(
                    dataset_info["images"]["url"],
                    str(images_zip),
                    expected_size_mb=dataset_info["images"]["size_gb"] * 1024
                )

            logger.info("Extracting images...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        if val_dir.exists() and not images_dir.exists():
            logger.info("Resizing images to 1024x1024 for SDXL benchmark...")
            _resize_coco_images(val_dir, images_dir, prompts_file, target_size=1024)

    fid_stats_file = data_dir / "val2014.npz"
    # MLCommons expects latents/latents.pt (subdirectory)
    latents_dir = data_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    latents_file = latents_dir / "latents.pt"
    mlcommons_captions = data_dir / "captions_source.tsv"

    # Required for MLCommons-compliant FID computation
    if "fid_statistics" in dataset_info:
        if not fid_stats_file.exists() or force:
            logger.info("Downloading MLCommons FID statistics (val2014.npz)...")
            try:
                _download_file(
                    dataset_info["fid_statistics"]["url"],
                    str(fid_stats_file),
                    expected_size_mb=dataset_info["fid_statistics"]["size_mb"]
                )
            except Exception as e:
                logger.warning(f"Failed to download FID statistics: {e}")
                logger.warning("FID computation will use reference images instead (not MLCommons-compliant)")

    # Required for reproducibility in closed division
    # MLCommons latents.pt is a single tensor (1, 4, 128, 128) ~256KB, shared by all samples
    if "latents" in dataset_info:
        if not latents_file.exists() or force:
            logger.info("Downloading MLCommons pre-generated latents...")
            try:
                _download_file(
                    dataset_info["latents"]["url"],
                    str(latents_file),
                    expected_size_mb=dataset_info["latents"]["size_mb"]
                )
            except Exception as e:
                logger.warning(f"Failed to download latents: {e}")
                logger.warning("Latents will be generated locally at first run (seed=0)")

    # For closed division, the EXACT MLCommons prompts must be used
    if "captions_tsv" in dataset_info:
        if not mlcommons_captions.exists() or force:
            logger.info("Downloading MLCommons official captions file...")
            try:
                _download_file(
                    dataset_info["captions_tsv"]["url"],
                    str(mlcommons_captions),
                    expected_size_mb=dataset_info["captions_tsv"]["size_mb"]
                )
            except Exception as e:
                logger.warning(f"Failed to download MLCommons captions: {e}")
                logger.warning("Using locally generated captions file")

        # Convert captions_source.tsv (7 columns) to simple format (id<tab>caption)
        # captions_source.tsv: id, image_id, caption, height, width, file_name, coco_url
        if mlcommons_captions.exists():
            count = 0
            with open(mlcommons_captions, 'r', encoding='utf-8') as src, \
                 open(prompts_file, 'w', encoding='utf-8') as dst:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if parts[0] == 'id':
                        continue
                    if len(parts) >= 3:
                        # Use image_id (col 1) and caption (col 2)
                        image_id = parts[1]
                        caption = parts[2]
                        dst.write(f"{image_id}\t{caption}\n")
                        count += 1
            logger.info(
                f"Converted MLCommons captions_source.tsv to {prompts_file} "
                f"({count} prompts)"
            )

    actual_samples = sum(1 for _ in open(prompts_file, 'r'))

    logger.info(f"COCO 2014 dataset ready at {data_dir}")

    return {
        "data_path": str(data_dir),
        "prompts_file": str(prompts_file),
        "captions_file": str(captions_file),
        "images_dir": str(images_dir) if images_dir.exists() else None,
        "fid_statistics": str(fid_stats_file) if fid_stats_file.exists() else None,
        "latents_file": str(latents_file) if latents_file.exists() else None,
        "num_samples": actual_samples,
    }


def download_cnn_dailymail(
    output_dir: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    force: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, str]:
    """Download and process CNN-DailyMail for MLPerf Llama 3.1 8B benchmark.

    Follows the MLCommons Inference v5.1 reference (download_cnndm.py) exactly:
    1. Downloads cnn_dailymail v3.0.0 validation split from HuggingFace
    2. Applies instruction template (plain text, NOT chat template)
    3. Pre-tokenizes inputs with tokenizer.encode() (adds BOS implicitly)
    4. NO token length filtering — all samples included (closed division)
    5. Saves as JSON with fields: instruction, input, tok_input, output

    Datacenter: 13,368 samples | Edge: 5,000 samples (shuffle seed=42)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "cnn-dailymail"
    data_dir.mkdir(parents=True, exist_ok=True)

    eval_file = data_dir / "cnn_eval.json"

    if eval_file.exists() and not force:
        import json

        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        actual_count = len(data) if isinstance(data, list) else 0
        logger.info(f"CNN-DailyMail dataset already exists: {actual_count} samples at {eval_file}")
        return {
            "data_path": str(data_dir),
            "eval_file": str(eval_file),
            "num_samples": actual_count,
        }

    logger.info("Downloading CNN-DailyMail v3.0.0 for MLPerf Llama 3.1 8B benchmark...")
    logger.info(f"Tokenizer model: {model_name}")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "datasets and transformers are required for CNN-DailyMail download. "
            "Install with: pip install datasets transformers"
        )

    logger.info("Loading cnn_dailymail v3.0.0 from HuggingFace...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")

    from .model_downloader import _resolve_hf_token
    token = _resolve_hf_token(hf_token)
    if not token:
        logger.warning(
            "No HuggingFace token found. Meta-Llama tokenizer requires authentication. "
            "Set HF_TOKEN env var or run: huggingface-cli login"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=token)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 8000

    INSTRUCTION_TEMPLATE = (
        "Summarize the following news article in 128 tokens. "
        "Please output the summary only, without any other text.\n\n"
        "Article:\n{input}\n\nSummary:"
    )

    logger.info("Processing articles with MLCommons instruction template...")
    processed = []
    for i, entry in enumerate(dataset):
        article = entry.get("article", "")
        highlights = entry.get("highlights", "")

        if not article or not highlights:
            continue

        formatted_prompt = INSTRUCTION_TEMPLATE.format(input=article)
        tok_input = tokenizer.encode(formatted_prompt)

        processed.append({
            "instruction": {"llama": INSTRUCTION_TEMPLATE},
            "input": article,
            "tok_input": tok_input,
            "output": highlights,
        })

        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i + 1}/{len(dataset)}, total: {len(processed)}")

    logger.info(f"Total samples: {len(processed)} (no filtering, closed division)")

    import json
    import random

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False)
    logger.info(f"Datacenter eval set: {len(processed)} samples -> {eval_file}")

    random.seed(42)
    n_edge = min(5000, len(processed))
    edge_indices = random.sample(range(len(processed)), n_edge)
    edge_samples = [processed[i] for i in sorted(edge_indices)]
    edge_file = data_dir / "sample_cnn_eval_5000.json"
    with open(edge_file, "w", encoding="utf-8") as f:
        json.dump(edge_samples, f, ensure_ascii=False)
    logger.info(f"Edge eval set: {len(edge_samples)} samples -> {edge_file}")

    calib_indices = random.sample(range(len(processed)), min(1000, len(processed)))
    calib_samples = [processed[i] for i in sorted(calib_indices)]
    calib_file = data_dir / "cnn_dailymail_calibration.json"
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_samples, f, ensure_ascii=False)
    logger.info(f"Calibration set: {len(calib_samples)} samples -> {calib_file}")

    return {
        "data_path": str(data_dir),
        "eval_file": str(eval_file),
        "num_samples": len(processed),
    }


def download_dataset(
    dataset_name: str,
    output_dir: str,
    subset: Optional[str] = None,
    force: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, str]:
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
    elif dataset_name == "coco2014":
        return download_coco2014(output_dir, force)
    elif dataset_name == "cnn-dailymail":
        return download_cnn_dailymail(output_dir, force=force, hf_token=hf_token)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _is_valid_zip(path: Path) -> bool:
    """Check if a file is a valid ZIP archive."""
    import zipfile
    if not path.exists():
        return False
    if path.stat().st_size < 100:
        return False
    try:
        with zipfile.ZipFile(str(path), 'r') as zf:
            # testzip() returns the name of the first bad file, or None if OK
            return zf.testzip() is None
    except (zipfile.BadZipFile, Exception):
        return False


def _download_and_extract_zip(
    url: str,
    zip_path: Path,
    extract_dir: Path,
    description: str,
    expected_size_mb: float,
    force: bool = False,
) -> None:
    """Download a ZIP file and extract it, with validation and auto-retry."""
    import zipfile

    need_download = force or not zip_path.exists()

    # If file exists but is not a valid ZIP, delete and re-download
    if zip_path.exists() and not need_download:
        if not _is_valid_zip(zip_path):
            logger.warning(f"Existing {zip_path.name} is corrupted, re-downloading...")
            zip_path.unlink()
            need_download = True

    if need_download:
        logger.info(f"Downloading {description} (~{expected_size_mb:.0f}MB)...")
        # Expect at least 1MB for any real dataset file
        _download_file(url, str(zip_path), expected_size_mb=expected_size_mb,
                        min_size_bytes=1024 * 1024)

        # Validate the download is a valid ZIP
        if not _is_valid_zip(zip_path):
            size_kb = zip_path.stat().st_size / 1024
            zip_path.unlink()
            raise RuntimeError(
                f"Downloaded file {zip_path.name} ({size_kb:.0f}KB) is not a valid ZIP archive. "
                f"The URL may have returned an error page. Check your network connection and "
                f"verify the URL is accessible: {url}"
            )

    logger.info(f"Extracting {description}...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_dir))
    logger.info(f"Extracted to {extract_dir}")


def download_coco2017(
    output_dir: str,
    force: bool = False,
) -> Dict[str, str]:
    """Download COCO 2017 validation dataset for SSD-ResNet34."""
    registry = DATASET_REGISTRY["coco2017"]

    output_path = Path(output_dir) / "coco2017"
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "val2017"
    annotations_dir = output_path / "annotations"

    # Download and extract images
    if not images_dir.exists() or force:
        img_info = registry["images"]
        zip_path = output_path / img_info["filename"]
        _download_and_extract_zip(
            url=img_info["url"],
            zip_path=zip_path,
            extract_dir=output_path,
            description="COCO 2017 validation images",
            expected_size_mb=1024,
            force=force,
        )
    else:
        num_images = len(list(images_dir.glob("*.jpg")))
        logger.info(f"COCO 2017 images already exist: {num_images} images")

    # Download and extract annotations
    if not annotations_dir.exists() or not (annotations_dir / "instances_val2017.json").exists() or force:
        ann_info = registry["annotations"]
        zip_path = output_path / ann_info["filename"]
        _download_and_extract_zip(
            url=ann_info["url"],
            zip_path=zip_path,
            extract_dir=output_path,
            description="COCO 2017 annotations",
            expected_size_mb=252,
            force=force,
        )
    else:
        logger.info("COCO 2017 annotations already exist")

    num_images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    logger.info(f"COCO 2017 dataset ready: {num_images} images")

    return {
        "data_path": str(output_path),
        "images_dir": str(images_dir),
        "annotations_file": str(annotations_dir / "instances_val2017.json"),
        "num_samples": num_images,
    }


def list_available_datasets() -> Dict[str, str]:
    return {
        "imagenet": DATASET_REGISTRY["imagenet"]["description"],
        "squad": DATASET_REGISTRY["squad"]["description"],
        "openimages": DATASET_REGISTRY["openimages"]["description"],
        "librispeech": DATASET_REGISTRY["librispeech"]["description"],
        "coco2014": DATASET_REGISTRY["coco2014"]["description"],
        "coco2017": DATASET_REGISTRY["coco2017"]["description"],
        "cnn-dailymail": DATASET_REGISTRY["cnn-dailymail"]["description"],
    }


def get_dataset_info(dataset_name: str) -> Dict:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name]


def redownload_corrupted_openimages(data_path: str, num_workers: int = 8) -> int:
    from PIL import Image

    data_dir = Path(data_path)
    images_dir = data_dir / "validation" / "data"
    if not images_dir.exists():
        images_dir = data_dir / "images"

    if not images_dir.exists():
        logger.error(f"Images directory not found in {data_path}")
        return 0

    corrupted = []
    logger.info("Checking for corrupted images...")

    for img_path in images_dir.glob("*.jpg"):
        try:
            if img_path.stat().st_size < 1000:
                corrupted.append(img_path.stem)
                continue

            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted.append(img_path.stem)

    if not corrupted:
        logger.info("No corrupted images found")
        return 0

    logger.info(f"Found {len(corrupted)} corrupted images, re-downloading...")

    for image_id in corrupted:
        img_path = images_dir / f"{image_id}.jpg"
        if img_path.exists():
            img_path.unlink()

    _download_openimages_from_s3(corrupted, images_dir, num_workers)

    cache_dir = data_dir / "preprocessed_cache"
    if cache_dir.exists():
        for image_id in corrupted:
            cache_file = cache_dir / f"{image_id}.npy"
            if cache_file.exists():
                cache_file.unlink()
        logger.info(f"Cleared {len(corrupted)} cached preprocessed files")

    return len(corrupted)
