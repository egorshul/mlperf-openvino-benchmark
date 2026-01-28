"""
Model downloader utility for MLPerf OpenVINO Benchmark.
"""

import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError

logger = logging.getLogger(__name__)


# Model download URLs and checksums
MODEL_REGISTRY: Dict[str, Dict] = {
    "resnet50": {
        "onnx": {
            "url": "https://zenodo.org/record/4735647/files/resnet50_v1.onnx",
            "filename": "resnet50_v1.onnx",
            "md5": None,  # Add checksum if available
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
    """
    Download a file from URL.
    
    Args:
        url: Source URL
        destination: Destination file path
        show_progress: Show download progress
    """
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
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, destination)
        
        logger.info(f"Downloaded to {destination}")
        
    except URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def _verify_checksum(file_path: str, expected_md5: str) -> bool:
    """
    Verify file checksum.
    
    Args:
        file_path: Path to file
        expected_md5: Expected MD5 hash
        
    Returns:
        True if checksum matches
    """
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
    """
    Download a model.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50')
        output_dir: Directory to save the model
        format: Model format ('onnx' or 'openvino')
        force: Force re-download even if file exists
        
    Returns:
        Path to downloaded model file
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_info = MODEL_REGISTRY[model_name]
    
    if format not in model_info:
        raise ValueError(f"Format '{format}' not available for {model_name}")
    
    format_info = model_info[format]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Destination file path
    dest_file = output_path / format_info["filename"]
    
    # Check if already downloaded
    if dest_file.exists() and not force:
        logger.info(f"Model already exists: {dest_file}")
        
        # Verify checksum if available
        if format_info.get("md5"):
            if _verify_checksum(str(dest_file), format_info["md5"]):
                return str(dest_file)
            else:
                logger.warning("Checksum mismatch, re-downloading...")
        else:
            return str(dest_file)
    
    # Download
    temp_file = str(dest_file) + ".tmp"
    try:
        _download_file(format_info["url"], temp_file)
        
        # Verify checksum
        if format_info.get("md5"):
            if not _verify_checksum(temp_file, format_info["md5"]):
                raise RuntimeError("Downloaded file checksum mismatch")
        
        # Move to final destination
        shutil.move(temp_file, str(dest_file))
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise
    
    return str(dest_file)


def convert_to_openvino(
    onnx_path: str,
    output_dir: str,
    compress_to_fp16: bool = True
) -> str:
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save IR model
        compress_to_fp16: Whether to compress weights to FP16
        
    Returns:
        Path to OpenVINO IR model (.xml file)
    """
    try:
        import openvino as ov
    except ImportError:
        raise ImportError("OpenVINO is required for model conversion")
    
    logger.info(f"Converting {onnx_path} to OpenVINO IR...")
    
    # Read ONNX model
    model = ov.Core().read_model(onnx_path)
    
    # Output path
    onnx_name = Path(onnx_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    xml_path = output_path / f"{onnx_name}.xml"
    
    # Save model
    ov.save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)
    
    logger.info(f"Converted model saved to {xml_path}")
    
    return str(xml_path)


def list_available_models() -> Dict[str, str]:
    """
    List available models for download.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        name: info.get("description", "No description")
        for name, info in MODEL_REGISTRY.items()
    }


def download_whisper_model(
    output_dir: str,
    model_id: str = "openai/whisper-large-v3",
    export_to_openvino: bool = True,
) -> Dict[str, str]:
    """
    Download and optionally export Whisper model to OpenVINO format.
    
    Args:
        output_dir: Directory to save the model
        model_id: HuggingFace model ID
        export_to_openvino: Whether to export to OpenVINO IR format
        
    Returns:
        Dictionary with paths to encoder and decoder models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = model_id.split("/")[-1]
    
    if export_to_openvino:
        return _export_whisper_to_openvino(output_dir, model_id)
    else:
        return _download_whisper_from_hf(output_dir, model_id)


def _download_whisper_from_hf(output_dir: str, model_id: str) -> Dict[str, str]:
    """
    Download Whisper model from HuggingFace.
    
    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID
        
    Returns:
        Dictionary with model paths
    """
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
    
    # Download model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    
    # Save locally
    model.save_pretrained(str(model_path))
    processor.save_pretrained(str(model_path))
    
    logger.info(f"Model saved to {model_path}")
    
    return {
        "model_path": str(model_path),
        "processor_path": str(model_path),
    }


def _configure_hf_download() -> None:
    """
    Configure HuggingFace Hub for reliable large file downloads.

    Sets timeouts and enables resume for interrupted downloads.
    """
    try:
        from huggingface_hub import constants
        import os

        # Enable resume downloads (default in newer versions)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

        # Set longer timeout for large files (5 minutes connection, 30 minutes read)
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

    except ImportError:
        pass


def _download_with_retry(
    download_func,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Any:
    """
    Execute download function with retry logic and exponential backoff.

    Args:
        download_func: Function to call for download
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Result from download_func
    """
    import time

    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return download_func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if it's a retryable error
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
            delay *= 2  # Exponential backoff

    raise RuntimeError(f"Download failed after {max_retries + 1} attempts: {last_error}")


def _export_whisper_to_openvino(output_dir: str, model_id: str) -> Dict[str, str]:
    """
    Export Whisper model to OpenVINO IR format with KV-cache support.

    This creates separate encoder, decoder, and decoder_with_past models
    for optimal performance with autoregressive decoding.
    Uses robust download with retry logic and timeout handling.

    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID

    Returns:
        Dictionary with paths to encoder, decoder, and decoder_with_past IR models
    """
    try:
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
    except ImportError:
        raise ImportError(
            "optimum-intel is required for Whisper export to OpenVINO. "
            "Install with: pip install optimum[openvino]"
        )

    # Configure HuggingFace Hub for reliable downloads
    _configure_hf_download()

    logger.info(f"Exporting Whisper model to OpenVINO (with KV-cache): {model_id}")

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino"

    # Check if already exported (must have encoder, decoder, and decoder_with_past)
    if ov_model_path.exists():
        encoder_path = ov_model_path / "encoder_model.xml"
        decoder_path = ov_model_path / "decoder_model.xml"
        decoder_with_past_path = ov_model_path / "decoder_with_past_model.xml"

        # Check for all required models including decoder_with_past
        if encoder_path.exists() and decoder_path.exists() and decoder_with_past_path.exists():
            logger.info(f"OpenVINO model with KV-cache already exists at {ov_model_path}")
            return {
                "encoder_path": str(encoder_path),
                "decoder_path": str(decoder_path),
                "decoder_with_past_path": str(decoder_with_past_path),
                "model_path": str(ov_model_path),
            }
        elif encoder_path.exists() and decoder_path.exists():
            # Model exists but without decoder_with_past - need to re-export
            logger.warning(
                f"Found model at {ov_model_path} but missing decoder_with_past_model.xml. "
                "Re-exporting for optimal KV-cache performance..."
            )
            # Remove old model to re-export with proper KV-cache support
            import shutil
            shutil.rmtree(str(ov_model_path))

    # Export model with retry logic
    logger.info("Downloading and exporting model with KV-cache support...")
    logger.info("This may take several minutes. Large files (3+ GB) - download will resume if interrupted.")

    def do_export():
        # OVModelForSpeechSeq2Seq automatically exports with decoder_with_past
        # when export=True for seq2seq models like Whisper
        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )

    model = _download_with_retry(do_export, max_retries=3)

    # Save model (this saves encoder, decoder, and decoder_with_past)
    model.save_pretrained(str(ov_model_path))

    # Also save processor with retry
    logger.info("Downloading processor...")

    def do_processor():
        return WhisperProcessor.from_pretrained(model_id)

    processor = _download_with_retry(do_processor, max_retries=3)
    processor.save_pretrained(str(ov_model_path))

    # Verify all models were created
    encoder_path = ov_model_path / "encoder_model.xml"
    decoder_path = ov_model_path / "decoder_model.xml"
    decoder_with_past_path = ov_model_path / "decoder_with_past_model.xml"

    if not encoder_path.exists():
        raise RuntimeError(f"Export failed: encoder_model.xml not found at {ov_model_path}")
    if not decoder_path.exists():
        raise RuntimeError(f"Export failed: decoder_model.xml not found at {ov_model_path}")
    if not decoder_with_past_path.exists():
        logger.warning(
            f"decoder_with_past_model.xml not created. "
            "The model will work but may have suboptimal performance without KV-cache."
        )

    logger.info(f"OpenVINO model saved to {ov_model_path}")
    logger.info(f"  - Encoder: {encoder_path.name}")
    logger.info(f"  - Decoder: {decoder_path.name}")
    if decoder_with_past_path.exists():
        logger.info(f"  - Decoder with KV-cache: {decoder_with_past_path.name}")

    result = {
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "model_path": str(ov_model_path),
    }

    if decoder_with_past_path.exists():
        result["decoder_with_past_path"] = str(decoder_with_past_path)

    return result


def export_whisper_encoder_only(
    output_dir: str,
    model_id: str = "openai/whisper-large-v3",
) -> str:
    """
    Export only the Whisper encoder to ONNX format.
    
    Useful for encoder-only benchmarking or using external decoder.
    
    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID
        
    Returns:
        Path to encoder ONNX model
    """
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
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    encoder = model.get_encoder()
    encoder.eval()
    
    # Create dummy input (batch=1, n_mels=80, time=3000 for 30 seconds)
    dummy_input = torch.randn(1, 80, 3000)
    
    # Export to ONNX
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
    """
    Download and optionally export SDXL model to OpenVINO format.

    Args:
        output_dir: Directory to save the model
        model_id: HuggingFace model ID
        export_to_openvino: Whether to export to OpenVINO IR format

    Returns:
        Dictionary with paths to model components
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = model_id.split("/")[-1]

    if export_to_openvino:
        return _export_sdxl_to_openvino(output_dir, model_id)
    else:
        return _download_sdxl_from_hf(output_dir, model_id)


def _download_sdxl_from_hf(output_dir: str, model_id: str) -> Dict[str, str]:
    """
    Download SDXL model from HuggingFace.

    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID

    Returns:
        Dictionary with model paths
    """
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

    # Download pipeline
    def do_download():
        return StableDiffusionXLPipeline.from_pretrained(
            model_id,
            use_safetensors=True,
        )

    pipeline = _download_with_retry(do_download, max_retries=3)

    # Save locally
    pipeline.save_pretrained(str(model_path))

    logger.info(f"Model saved to {model_path}")

    return {
        "model_path": str(model_path),
    }


def _export_sdxl_to_openvino(output_dir: str, model_id: str) -> Dict[str, str]:
    """
    Export SDXL model to OpenVINO IR format.

    Uses Optimum-Intel for optimal OpenVINO export with all components
    (text encoders, UNet, VAE).

    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID

    Returns:
        Dictionary with paths to OpenVINO IR models
    """
    try:
        from optimum.intel import OVStableDiffusionXLPipeline
    except ImportError:
        raise ImportError(
            "optimum-intel is required for SDXL export to OpenVINO. "
            "Install with: pip install optimum[openvino] diffusers"
        )

    # Configure HuggingFace Hub for reliable downloads
    _configure_hf_download()

    logger.info(f"Exporting SDXL model to OpenVINO: {model_id}")

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino"

    # Check if already exported
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

    # Export model with retry logic
    logger.info("Downloading and exporting SDXL model (this may take a long time)...")
    logger.info("Large files (6+ GB) - download will resume if interrupted.")

    def do_export():
        return OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )

    pipeline = _download_with_retry(do_export, max_retries=3)

    # Save model
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
    """
    Download MLPerf RetinaNet model and prepare with multiple batch sizes.

    Downloads the pre-converted ONNX model from Zenodo and creates
    OpenVINO IR models for different batch sizes (1, 2, 4, 8).

    Args:
        output_dir: Directory to save the models
        batch_sizes: List of batch sizes to create (default: [1, 2, 4, 8])
        convert_to_openvino: Whether to convert to OpenVINO IR format
        force: Force re-download even if files exist

    Returns:
        Dictionary with paths to models for each batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # MLPerf RetinaNet ONNX model from Zenodo
    onnx_url = "https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx"
    onnx_filename = "retinanet.onnx"
    onnx_path = output_path / onnx_filename

    # Download ONNX model if not exists
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

    # Convert to OpenVINO IR for each batch size
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

        # Read ONNX model
        model = core.read_model(str(onnx_path))

        # Set static batch size for input
        # RetinaNet input: [N, 3, 800, 800] - set N to batch_size
        for input_node in model.inputs:
            input_shape = input_node.get_partial_shape()
            if input_shape.rank.get_length() == 4:
                # Set batch dimension to static value
                input_shape[0] = batch_size
                model.reshape({input_node: input_shape})
                logger.info(f"    Input shape set to: {input_shape}")

        # Save model with FP16 compression
        ov.save_model(model, str(xml_path), compress_to_fp16=True)
        logger.info(f"    Saved to: {xml_path}")

        result["batch_models"][batch_size] = str(xml_path)

    # Set default model path (batch size 1)
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
    """
    Get path to RetinaNet model for a specific batch size.

    Args:
        models_dir: Directory containing models
        batch_size: Desired batch size

    Returns:
        Path to model file or None if not found
    """
    models_path = Path(models_dir)

    # Try batch-specific OpenVINO IR
    batch_xml = models_path / f"retinanet_bs{batch_size}" / f"retinanet_bs{batch_size}.xml"
    if batch_xml.exists():
        return str(batch_xml)

    # Fall back to ONNX
    onnx_path = models_path / "retinanet.onnx"
    if onnx_path.exists():
        return str(onnx_path)

    return None
