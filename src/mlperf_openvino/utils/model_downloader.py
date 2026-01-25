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
    Download and export Whisper model to OpenVINO format.

    NOTE: For optimal performance with optimum-intel, Whisper models
    are always exported to OpenVINO IR format (.xml/.bin), regardless
    of the export_to_openvino parameter. The parameter is kept for
    API compatibility.

    Args:
        output_dir: Directory to save the model
        model_id: HuggingFace model ID
        export_to_openvino: Ignored - always exports to OpenVINO format

    Returns:
        Dictionary with paths to encoder and decoder models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Always export to OpenVINO for proper optimum-intel support
    if not export_to_openvino:
        logger.info(
            "Note: Whisper models are always exported to OpenVINO format "
            "for optimal performance with optimum-intel."
        )

    return _export_whisper_to_openvino(output_dir, model_id)


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

    # Check if already exported - try different naming conventions
    if ov_model_path.exists():
        # Try different encoder names (optimum-intel versions vary)
        encoder_candidates = [
            ov_model_path / "encoder_model.xml",
            ov_model_path / "openvino_encoder_model.xml",
        ]
        # Try different decoder names
        decoder_candidates = [
            ov_model_path / "decoder_model.xml",
            ov_model_path / "openvino_decoder_model.xml",
            ov_model_path / "decoder_with_past_model.xml",
            ov_model_path / "openvino_decoder_with_past_model.xml",
            ov_model_path / "decoder_model_merged.xml",
            ov_model_path / "openvino_decoder_model_merged.xml",
        ]

        encoder_path = None
        decoder_path = None

        for ep in encoder_candidates:
            if ep.exists():
                encoder_path = ep
                break

        for dp in decoder_candidates:
            if dp.exists():
                decoder_path = dp
                break

        if encoder_path and decoder_path:
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            logger.info(f"  Encoder: {encoder_path.name}")
            logger.info(f"  Decoder: {decoder_path.name}")
            return {
                "encoder_path": str(encoder_path),
                "decoder_path": str(decoder_path),
                "model_path": str(ov_model_path),
            }

    # Export model using optimum-cli for proper KV-cache support
    logger.info("Downloading and exporting model (this may take several minutes)...")
    logger.info("Large files (3+ GB) - download will resume if interrupted.")

    # Try optimum-cli first (more reliable for stateful export)
    import subprocess
    import shutil

    optimum_cli = shutil.which("optimum-cli")
    export_success = False

    if optimum_cli:
        logger.info("Using optimum-cli for export with stateful decoder...")
        try:
            cmd = [
                optimum_cli, "export", "openvino",
                "--model", model_id,
                "--task", "automatic-speech-recognition-with-past",
                str(ov_model_path),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            if result.returncode == 0:
                export_success = True
                logger.info("optimum-cli export completed successfully")
            else:
                logger.warning(f"optimum-cli failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("optimum-cli export timed out")
        except Exception as e:
            logger.warning(f"optimum-cli export failed: {e}")

    # Fallback to Python API if CLI failed
    if not export_success:
        logger.info("Using Python API for export...")

        def do_export():
            # Try with use_cache=True for decoder_with_past
            return OVModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                export=True,
                compile=False,
                use_cache=True,  # Enable KV-cache for decoder
            )

        model = _download_with_retry(do_export, max_retries=3)
        model.save_pretrained(str(ov_model_path))

    # List exported files
    xml_files = list(ov_model_path.glob("*.xml"))
    logger.info(f"Exported model files: {[f.name for f in xml_files]}")

    # Save processor
    logger.info("Downloading processor...")

    def do_processor():
        return WhisperProcessor.from_pretrained(model_id)

    processor = _download_with_retry(do_processor, max_retries=3)
    processor.save_pretrained(str(ov_model_path))

    logger.info(f"OpenVINO model saved to {ov_model_path}")

    # Find actual file names after export
    encoder_path = None
    decoder_path = None

    for ep in [ov_model_path / "encoder_model.xml", ov_model_path / "openvino_encoder_model.xml"]:
        if ep.exists():
            encoder_path = ep
            break

    for dp in [
        ov_model_path / "decoder_model.xml",
        ov_model_path / "openvino_decoder_model.xml",
        ov_model_path / "decoder_with_past_model.xml",
        ov_model_path / "openvino_decoder_with_past_model.xml",
        ov_model_path / "decoder_model_merged.xml",
    ]:
        if dp.exists():
            decoder_path = dp
            break

    # List what was actually created
    xml_files = list(ov_model_path.glob("*.xml"))
    logger.info(f"Created OpenVINO IR files: {[f.name for f in xml_files]}")

    return {
        "encoder_path": str(encoder_path) if encoder_path else str(ov_model_path / "encoder_model.xml"),
        "decoder_path": str(decoder_path) if decoder_path else str(ov_model_path / "decoder_model.xml"),
        "model_path": str(ov_model_path),
    }


def export_whisper_for_npu(
    output_dir: str,
    model_id: str = "openai/whisper-large-v3",
    batch_size: int = 1,
    encoder_seq_len: int = 1500,
    decoder_seq_len: int = 448,
    stateless: bool = False,
) -> Dict[str, str]:
    """
    Export Whisper model with static shapes optimized for NPU.

    NPU devices typically require static shapes for optimal performance.
    This function exports the model with fixed input dimensions.

    For NPU devices that don't support stateful operations (ReadValue/Assign),
    use stateless=True to export with explicit KV-cache tensors as inputs/outputs.

    Args:
        output_dir: Directory to save the model
        model_id: HuggingFace model ID
        batch_size: Fixed batch size (default 1)
        encoder_seq_len: Encoder sequence length (1500 for 30s audio)
        decoder_seq_len: Max decoder sequence length (448 for Whisper)
        stateless: If True, export without stateful KV-cache (for NPU compatibility)

    Returns:
        Dictionary with paths to model files
    """
    import subprocess
    import shutil

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    suffix = "-openvino-npu-stateless" if stateless else "-openvino-npu"
    ov_model_path = output_path / f"{model_name}{suffix}"

    logger.info(f"Exporting Whisper for NPU:")
    logger.info(f"  batch_size={batch_size}")
    logger.info(f"  encoder_seq_len={encoder_seq_len}")
    logger.info(f"  decoder_seq_len={decoder_seq_len}")
    logger.info(f"  stateless={stateless}")

    # Check if already exported
    if ov_model_path.exists():
        encoder_path = ov_model_path / "openvino_encoder_model.xml"
        decoder_path = ov_model_path / "openvino_decoder_model.xml"
        decoder_with_past_path = ov_model_path / "openvino_decoder_with_past_model.xml"

        # Accept either decoder or decoder_with_past
        has_decoder = decoder_path.exists() or decoder_with_past_path.exists()

        if encoder_path.exists() and has_decoder:
            logger.info(f"NPU model already exists at {ov_model_path}")
            actual_decoder = decoder_path if decoder_path.exists() else decoder_with_past_path
            return {
                "encoder_path": str(encoder_path),
                "decoder_path": str(actual_decoder),
                "model_path": str(ov_model_path),
            }

    # Use optimum-cli with static shapes
    optimum_cli = shutil.which("optimum-cli")

    if not optimum_cli:
        raise RuntimeError(
            "optimum-cli is required for NPU export. "
            "Install with: pip install optimum[openvino]"
        )

    # IMPORTANT: Use automatic-speech-recognition-with-past for proper KV-cache support
    # Combined with --disable-stateful, this gives explicit KV-cache tensors instead of
    # internal ReadValue/Assign operations that some NPUs don't support
    task = "automatic-speech-recognition-with-past"

    cmd = [
        optimum_cli, "export", "openvino",
        "--model", model_id,
        "--task", task,
    ]

    # Add stateless flag if requested (converts KV-cache from internal state to explicit tensors)
    if stateless:
        cmd.append("--disable-stateful")
        logger.info("Exporting in STATELESS mode:")
        logger.info("  - No ReadValue/Assign operations (NPU compatible)")
        logger.info("  - KV-cache passed as explicit input/output tensors")
    else:
        logger.info("Exporting in STATEFUL mode (with ReadValue/Assign ops)")
        logger.info("NOTE: Some NPUs don't support stateful ops. Use --stateless if compilation fails.")

    cmd.append(str(ov_model_path))

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            logger.error(f"Export failed: {result.stderr}")
            raise RuntimeError(f"optimum-cli export failed: {result.stderr}")

        logger.info("Export completed successfully")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Export timed out after 30 minutes")

    # List exported files
    xml_files = list(ov_model_path.glob("*.xml"))
    logger.info(f"Exported model files: {[f.name for f in xml_files]}")

    # Analyze decoder for stateful ops
    _analyze_model_stateful_ops(ov_model_path)

    # Save processor
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(model_id)
        processor.save_pretrained(str(ov_model_path))
        logger.info("Processor saved")
    except Exception as e:
        logger.warning(f"Failed to save processor: {e}")

    # Find paths
    encoder_path = ov_model_path / "openvino_encoder_model.xml"
    decoder_path = ov_model_path / "openvino_decoder_model.xml"
    decoder_with_past_path = ov_model_path / "openvino_decoder_with_past_model.xml"

    actual_decoder = decoder_with_past_path if decoder_with_past_path.exists() else decoder_path

    return {
        "encoder_path": str(encoder_path) if encoder_path.exists() else None,
        "decoder_path": str(actual_decoder) if actual_decoder.exists() else None,
        "model_path": str(ov_model_path),
    }


def _analyze_model_stateful_ops(model_dir: Path) -> Dict[str, int]:
    """
    Analyze model files for stateful operations (ReadValue/Assign).

    Args:
        model_dir: Directory containing model XML files

    Returns:
        Dictionary mapping model name to count of stateful ops
    """
    try:
        import openvino as ov
        core = ov.Core()
    except ImportError:
        return {}

    results = {}

    for xml_file in model_dir.glob("*.xml"):
        try:
            model = core.read_model(xml_file)

            # Count stateful ops
            read_values = 0
            assigns = 0

            for op in model.get_ordered_ops():
                op_type = op.get_type_name()
                if op_type == "ReadValue":
                    read_values += 1
                elif op_type == "Assign":
                    assigns += 1

            total_stateful = read_values + assigns
            results[xml_file.name] = total_stateful

            if total_stateful > 0:
                logger.info(f"  {xml_file.name}: {read_values} ReadValue + {assigns} Assign = {total_stateful} stateful ops")
            else:
                logger.info(f"  {xml_file.name}: No stateful ops (NPU compatible)")

        except Exception as e:
            logger.warning(f"  Failed to analyze {xml_file.name}: {e}")

    return results


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
