"""
Model downloader utility for MLPerf OpenVINO Benchmark.
"""

import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional
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
) -> any:
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
    Export Whisper model to OpenVINO IR format.

    This creates separate encoder and decoder models for optimal performance.
    Uses robust download with retry logic and timeout handling.

    Args:
        output_dir: Output directory
        model_id: HuggingFace model ID

    Returns:
        Dictionary with paths to encoder and decoder IR models
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

    logger.info(f"Exporting Whisper model to OpenVINO: {model_id}")

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino"

    # Check if already exported
    if ov_model_path.exists():
        encoder_path = ov_model_path / "encoder_model.xml"
        decoder_path = ov_model_path / "decoder_model.xml"

        if encoder_path.exists() and decoder_path.exists():
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            return {
                "encoder_path": str(encoder_path),
                "decoder_path": str(decoder_path),
                "model_path": str(ov_model_path),
            }

    # Export model with retry logic
    logger.info("Downloading and exporting model (this may take several minutes)...")
    logger.info("Large files (3+ GB) - download will resume if interrupted.")

    def do_export():
        return OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )

    model = _download_with_retry(do_export, max_retries=3)

    # Save model
    model.save_pretrained(str(ov_model_path))

    # Also save processor with retry
    logger.info("Downloading processor...")

    def do_processor():
        return WhisperProcessor.from_pretrained(model_id)

    processor = _download_with_retry(do_processor, max_retries=3)
    processor.save_pretrained(str(ov_model_path))

    logger.info(f"OpenVINO model saved to {ov_model_path}")

    return {
        "encoder_path": str(ov_model_path / "encoder_model.xml"),
        "decoder_path": str(ov_model_path / "decoder_model.xml"),
        "model_path": str(ov_model_path),
    }


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
