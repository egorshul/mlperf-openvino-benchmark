import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError

logger = logging.getLogger(__name__)


MODEL_REGISTRY: Dict[str, Dict] = {
    "resnet50": {
        "onnx": {
            "url": "https://zenodo.org/record/4735647/files/resnet50_v1.onnx",
            "filename": "resnet50_v1.onnx",
            "md5": None,
        },
        "description": "ResNet50-v1.5 for ImageNet classification",
    },
    "bert": {
        "onnx": {
            "url": "https://zenodo.org/record/3733910/files/model.onnx",
            "filename": "bert_large.onnx",
            "md5": None,
        },
        "description": "BERT-Large for question answering",
    },
    "retinanet": {
        "onnx": {
            "url": "https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx",
            "filename": "retinanet.onnx",
            "md5": None,
        },
        "description": "RetinaNet for object detection",
    },
    "whisper": {
        "huggingface": {
            "model_id": "openai/whisper-large-v3",
            "filename": "whisper-large-v3",
        },
        "description": "Whisper Large v3 for speech recognition",
    },
    "sdxl": {
        "huggingface": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "filename": "stable-diffusion-xl-base-1.0",
        },
        "description": "Stable Diffusion XL 1.0 for text-to-image generation",
    },
}


def _download_file(url: str, destination: str, show_progress: bool = True) -> None:
    logger.info(f"Downloading from {url}")

    try:
        if show_progress:
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded / total_size) * 100)
                    print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)",
                          end="", flush=True)

            urllib.request.urlretrieve(url, destination, progress_hook)
            print()
        else:
            urllib.request.urlretrieve(url, destination)

        logger.info(f"Downloaded to {destination}")

    except URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def _verify_checksum(file_path: str, expected_md5: str) -> bool:
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


def download_model(
    model_name: str,
    output_dir: str,
    format: str = "onnx",
    force: bool = False
) -> str:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_info = MODEL_REGISTRY[model_name]

    if format not in model_info:
        raise ValueError(f"Format '{format}' not available for {model_name}")

    format_info = model_info[format]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dest_file = output_path / format_info["filename"]

    if dest_file.exists() and not force:
        logger.info(f"Model already exists: {dest_file}")

        if format_info.get("md5"):
            if _verify_checksum(str(dest_file), format_info["md5"]):
                return str(dest_file)
            else:
                logger.warning("Checksum mismatch, re-downloading...")
        else:
            return str(dest_file)

    temp_file = str(dest_file) + ".tmp"
    try:
        _download_file(format_info["url"], temp_file)

        if format_info.get("md5"):
            if not _verify_checksum(temp_file, format_info["md5"]):
                raise RuntimeError("Downloaded file checksum mismatch")

        shutil.move(temp_file, str(dest_file))

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

    return str(dest_file)


def convert_to_openvino(
    onnx_path: str,
    output_dir: str,
    compress_to_fp16: bool = True
) -> str:
    try:
        import openvino as ov
    except ImportError:
        raise ImportError("OpenVINO is required for model conversion")

    logger.info(f"Converting {onnx_path} to OpenVINO IR...")

    model = ov.Core().read_model(onnx_path)

    onnx_name = Path(onnx_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_path = output_path / f"{onnx_name}.xml"

    ov.save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)

    logger.info(f"Converted model saved to {xml_path}")

    return str(xml_path)


def list_available_models() -> Dict[str, str]:
    return {
        name: info.get("description", "No description")
        for name, info in MODEL_REGISTRY.items()
    }


def download_whisper_model(
    output_dir: str,
    model_id: str = "openai/whisper-large-v3",
    export_to_openvino: bool = True,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = model_id.split("/")[-1]

    if export_to_openvino:
        return _export_whisper_to_openvino(output_dir, model_id)
    else:
        return _download_whisper_from_hf(output_dir, model_id)


def _download_whisper_from_hf(output_dir: str, model_id: str) -> Dict[str, str]:
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        raise ImportError(
            "transformers is required for Whisper download. "
            "Install with: pip install transformers"
        )

    logger.info(f"Downloading Whisper model: {model_id}")

    output_path = Path(output_dir)
    model_path = output_path / model_id.split("/")[-1]

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)

    model.save_pretrained(str(model_path))
    processor.save_pretrained(str(model_path))

    logger.info(f"Model saved to {model_path}")

    return {
        "model_path": str(model_path),
        "processor_path": str(model_path),
    }


def _configure_hf_download() -> None:
    """Configure HuggingFace Hub for reliable large file downloads."""
    try:
        from huggingface_hub import constants
        import os

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        # 30 minute timeout for large files
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

    except ImportError:
        pass


def _download_with_retry(
    download_func,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Any:
    import time

    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return download_func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            retryable = any(x in error_str for x in [
                "timeout", "connection", "network", "reset",
                "incomplete", "aborted", "refused"
            ])

            if not retryable or attempt == max_retries:
                break

            logger.warning(
                f"Download attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Download failed after {max_retries + 1} attempts: {last_error}")


def _find_openvino_model(model_dir: Path, base_name: str) -> Optional[Path]:
    candidates = [
        model_dir / f"openvino_{base_name}.xml",
        model_dir / f"{base_name}.xml",
    ]

    # decoder_with_past may also be stored as a merged decoder
    if base_name == "decoder_with_past_model":
        candidates.extend([
            model_dir / "openvino_decoder_model_merged.xml",
            model_dir / "decoder_model_merged.xml",
        ])

    for path in candidates:
        if path.exists():
            return path

    return None


def _export_whisper_to_openvino(output_dir: str, model_id: str) -> Dict[str, str]:
    try:
        from optimum.exporters.openvino import main_export
        from transformers import WhisperProcessor
    except ImportError:
        raise ImportError(
            "optimum-intel is required. Install with: pip install optimum[openvino]"
        )

    _configure_hf_download()

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino"

    if ov_model_path.exists():
        encoder_path = _find_openvino_model(ov_model_path, "encoder_model")
        decoder_path = _find_openvino_model(ov_model_path, "decoder_model")

        if encoder_path and decoder_path:
            logger.info(f"Model already exists at {ov_model_path}")
            result = {
                "encoder_path": str(encoder_path),
                "decoder_path": str(decoder_path),
                "model_path": str(ov_model_path),
            }
            decoder_with_past = _find_openvino_model(ov_model_path, "decoder_with_past_model")
            if decoder_with_past:
                result["decoder_with_past_path"] = str(decoder_with_past)
            return result
        else:
            shutil.rmtree(str(ov_model_path))

    logger.info(f"Exporting {model_id} to OpenVINO IR...")

    main_export(
        model_name_or_path=model_id,
        output=str(ov_model_path),
        task="automatic-speech-recognition-with-past",
    )

    processor = WhisperProcessor.from_pretrained(model_id)
    processor.save_pretrained(str(ov_model_path))

    encoder_path = _find_openvino_model(ov_model_path, "encoder_model")
    decoder_path = _find_openvino_model(ov_model_path, "decoder_model")
    decoder_with_past_path = _find_openvino_model(ov_model_path, "decoder_with_past_model")

    if not encoder_path or not decoder_path:
        created_files = list(ov_model_path.glob("*.xml"))
        raise RuntimeError(
            f"Export failed. Created files: {[f.name for f in created_files]}"
        )

    logger.info(f"Model exported to {ov_model_path}")

    result = {
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "model_path": str(ov_model_path),
    }
    if decoder_with_past_path:
        result["decoder_with_past_path"] = str(decoder_with_past_path)

    return result


def export_whisper_encoder_only(
    output_dir: str,
    model_id: str = "openai/whisper-large-v3",
) -> str:
    try:
        import torch
        from transformers import WhisperForConditionalGeneration
    except ImportError:
        raise ImportError(
            "torch and transformers are required. "
            "Install with: pip install torch transformers"
        )

    logger.info(f"Exporting Whisper encoder: {model_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = model_id.split("/")[-1]
    encoder_path = output_path / f"{model_name}_encoder.onnx"

    if encoder_path.exists():
        logger.info(f"Encoder already exists: {encoder_path}")
        return str(encoder_path)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    encoder = model.get_encoder()
    encoder.eval()

    # batch=1, n_mels=80, time=3000 (30 seconds)
    dummy_input = torch.randn(1, 80, 3000)

    logger.info(f"Exporting to {encoder_path}...")

    torch.onnx.export(
        encoder,
        dummy_input,
        str(encoder_path),
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_features": {0: "batch_size", 2: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    logger.info(f"Encoder exported to {encoder_path}")

    return str(encoder_path)


def download_sdxl_model(
    output_dir: str,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    export_to_openvino: bool = True,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = model_id.split("/")[-1]

    if export_to_openvino:
        return _export_sdxl_to_openvino(output_dir, model_id)
    else:
        return _download_sdxl_from_hf(output_dir, model_id)


def _download_sdxl_from_hf(output_dir: str, model_id: str) -> Dict[str, str]:
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        raise ImportError(
            "diffusers is required for SDXL download. "
            "Install with: pip install diffusers"
        )

    logger.info(f"Downloading SDXL model: {model_id}")

    output_path = Path(output_dir)
    model_path = output_path / model_id.split("/")[-1]

    def do_download():
        return StableDiffusionXLPipeline.from_pretrained(
            model_id,
            use_safetensors=True,
        )

    pipeline = _download_with_retry(do_download, max_retries=3)

    pipeline.save_pretrained(str(model_path))

    logger.info(f"Model saved to {model_path}")

    return {
        "model_path": str(model_path),
    }


def _export_sdxl_to_openvino(output_dir: str, model_id: str) -> Dict[str, str]:
    try:
        from optimum.intel import OVStableDiffusionXLPipeline
    except ImportError:
        raise ImportError(
            "optimum-intel is required for SDXL export to OpenVINO. "
            "Install with: pip install optimum[openvino] diffusers"
        )

    _configure_hf_download()

    logger.info(f"Exporting SDXL model to OpenVINO: {model_id}")

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino"

    if ov_model_path.exists():
        unet_path = ov_model_path / "unet" / "openvino_model.xml"
        vae_path = ov_model_path / "vae_decoder" / "openvino_model.xml"

        if unet_path.exists() and vae_path.exists():
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            return {
                "model_path": str(ov_model_path),
                "unet_path": str(unet_path),
                "vae_decoder_path": str(vae_path),
            }

    logger.info("Downloading and exporting SDXL model (this may take a long time)...")
    logger.info("Large files (6+ GB) - download will resume if interrupted.")

    def do_export():
        # load_in_8bit=False avoids NNCF int8 compression which degrades
        # accuracy for MLCommons closed division
        return OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            export=True,
            compile=False,
            load_in_8bit=False,
        )

    pipeline = _download_with_retry(do_export, max_retries=3)

    pipeline.save_pretrained(str(ov_model_path))

    logger.info(f"OpenVINO model saved to {ov_model_path}")

    return {
        "model_path": str(ov_model_path),
        "unet_path": str(ov_model_path / "unet" / "openvino_model.xml"),
        "vae_decoder_path": str(ov_model_path / "vae_decoder" / "openvino_model.xml"),
        "text_encoder_path": str(ov_model_path / "text_encoder" / "openvino_model.xml"),
        "text_encoder_2_path": str(ov_model_path / "text_encoder_2" / "openvino_model.xml"),
    }


def download_retinanet_model(
    output_dir: str,
    batch_sizes: list = None,
    convert_to_openvino: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_url = "https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx"
    onnx_filename = "retinanet.onnx"
    onnx_path = output_path / onnx_filename

    if not onnx_path.exists() or force:
        logger.info(f"Downloading RetinaNet ONNX model from Zenodo...")
        temp_file = str(onnx_path) + ".tmp"
        try:
            _download_file(onnx_url, temp_file)
            shutil.move(temp_file, str(onnx_path))
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise RuntimeError(f"Failed to download RetinaNet model: {e}")
    else:
        logger.info(f"RetinaNet ONNX model already exists: {onnx_path}")

    result = {
        "onnx_path": str(onnx_path),
        "batch_models": {},
    }

    if not convert_to_openvino:
        return result

    logger.info(f"Converting RetinaNet to OpenVINO IR with batch sizes: {batch_sizes}")

    try:
        import openvino as ov
    except ImportError:
        raise ImportError("OpenVINO is required for model conversion")

    core = ov.Core()

    for batch_size in batch_sizes:
        batch_dir = output_path / f"retinanet_bs{batch_size}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        xml_path = batch_dir / f"retinanet_bs{batch_size}.xml"

        if xml_path.exists() and not force:
            logger.info(f"  Batch size {batch_size}: already exists at {xml_path}")
            result["batch_models"][batch_size] = str(xml_path)
            continue

        logger.info(f"  Converting batch size {batch_size}...")

        model = core.read_model(str(onnx_path))

        # RetinaNet input: [N, 3, 800, 800] - set N to static batch_size
        for input_node in model.inputs:
            input_shape = input_node.get_partial_shape()
            if input_shape.rank.get_length() == 4:
                input_shape[0] = batch_size
                model.reshape({input_node: input_shape})
                logger.info(f"    Input shape set to: {input_shape}")

        ov.save_model(model, str(xml_path), compress_to_fp16=True)
        logger.info(f"    Saved to: {xml_path}")

        result["batch_models"][batch_size] = str(xml_path)

    if 1 in result["batch_models"]:
        result["model_path"] = result["batch_models"][1]
    elif result["batch_models"]:
        result["model_path"] = list(result["batch_models"].values())[0]

    logger.info(f"RetinaNet models ready:")
    for bs, path in result["batch_models"].items():
        logger.info(f"  Batch {bs}: {path}")

    return result


def get_retinanet_model_path(
    models_dir: str,
    batch_size: int = 1,
) -> Optional[str]:
    models_path = Path(models_dir)

    batch_xml = models_path / f"retinanet_bs{batch_size}" / f"retinanet_bs{batch_size}.xml"
    if batch_xml.exists():
        return str(batch_xml)

    onnx_path = models_path / "retinanet.onnx"
    if onnx_path.exists():
        return str(onnx_path)

    return None
