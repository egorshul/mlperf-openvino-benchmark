import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError

logger = logging.getLogger(__name__)


def _resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """Resolve HuggingFace authentication token.

    Checks (in order):
    1. Explicitly passed token argument
    2. HF_TOKEN environment variable
    3. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
    4. Cached token from `huggingface-cli login`

    Returns None if no token is found.
    """
    if token:
        return token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    return None


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
    "ssd-resnet34": {
        "onnx": {
            "url": "https://zenodo.org/record/4735664/files/ssd_resnet34_mAP_20.2.onnx",
            "filename": "ssd_resnet34_mAP_20.2.onnx",
            "md5": None,
        },
        "description": "SSD-ResNet34 for object detection (COCO 2017)",
    },
    "sdxl": {
        "huggingface": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "filename": "stable-diffusion-xl-base-1.0",
        },
        "description": "Stable Diffusion XL 1.0 for text-to-image generation",
    },
    "llama3.1-8b": {
        "huggingface": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "filename": "Llama-3.1-8B-Instruct",
        },
        "description": "Meta Llama 3.1 8B Instruct for text generation (MLPerf v5.1)",
    },
    "llama2-70b": {
        "huggingface": {
            "model_id": "meta-llama/Llama-2-70b-chat-hf",
            "filename": "Llama-2-70b-chat-hf",
        },
        "description": "Meta Llama 2 70B Chat for text generation (MLPerf Inference)",
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

    if base_name == "decoder_with_past_model":
        candidates.extend([
            model_dir / "openvino_decoder_model_merged.xml",
            model_dir / "decoder_model_merged.xml",
        ])

    for path in candidates:
        if path.exists():
            return path

    return None


def _convert_tokenizer_to_openvino(
    ov_model_path: Path,
    model_id: str,
    token: Optional[str] = None,
) -> Dict[str, str]:
    """Convert HuggingFace tokenizer to OpenVINO tokenizer/detokenizer models.

    This bypasses optimum-intel's ``maybe_convert_tokenizers()`` which fails
    when the externally-provided openvino-tokenizers reports a non-PEP-440
    version string (e.g. ``-1-85be884``).  Instead we call
    ``openvino_tokenizers.convert_tokenizer`` directly.

    Args:
        ov_model_path: Directory containing the exported OpenVINO model.
        model_id: HuggingFace model ID (e.g. "openai/whisper-large-v3").
        token: Optional HuggingFace access token for gated models.

    Returns:
        Dict with tokenizer_path and detokenizer_path if successful,
        empty dict otherwise.
    """
    result: Dict[str, str] = {}

    tokenizer_xml = ov_model_path / "openvino_tokenizer.xml"
    detokenizer_xml = ov_model_path / "openvino_detokenizer.xml"

    if tokenizer_xml.exists() and detokenizer_xml.exists():
        logger.info("OpenVINO tokenizer/detokenizer already exist")
        result["tokenizer_path"] = str(tokenizer_xml)
        result["detokenizer_path"] = str(detokenizer_xml)
        return result

    try:
        import openvino as ov
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.warning(
            f"Cannot convert tokenizer to OpenVINO ({e}). "
            "Install with: pip install --no-deps openvino-tokenizers"
        )
        return result

    logger.info("Converting tokenizer to OpenVINO format...")

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, token=token,
    )
    if not getattr(hf_tokenizer, "is_fast", False):
        logger.warning(
            "Fast tokenizer not available for %s. "
            "OpenVINO tokenizer conversion requires a fast tokenizer. "
            "Skipping tokenizer conversion.",
            model_id,
        )
        return result

    try:
        ov_tokenizer, ov_detokenizer = convert_tokenizer(
            hf_tokenizer, with_detokenizer=True
        )
        ov.save_model(ov_tokenizer, str(tokenizer_xml))
        ov.save_model(ov_detokenizer, str(detokenizer_xml))
        logger.info(f"Tokenizer saved to {tokenizer_xml}")
        logger.info(f"Detokenizer saved to {detokenizer_xml}")
        result["tokenizer_path"] = str(tokenizer_xml)
        result["detokenizer_path"] = str(detokenizer_xml)
    except Exception as e:
        logger.warning(f"Failed to convert tokenizer to OpenVINO: {e}")

    return result


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
            tok_result = _convert_tokenizer_to_openvino(ov_model_path, model_id)
            result.update(tok_result)
            return result
        else:
            shutil.rmtree(str(ov_model_path))

    logger.info(f"Exporting {model_id} to OpenVINO IR...")

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
        warnings.filterwarnings("ignore", message=".*Output nr.*does not match.*")
        warnings.filterwarnings("ignore", message=".*traced function does not match.*")
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        warnings.filterwarnings("ignore", message=".*use_fast.*")
        warnings.filterwarnings("ignore", message=".*loss_type.*")
        warnings.filterwarnings("ignore", message=".*Moving the following attributes.*generation config.*")
        try:
            from torch.jit import TracerWarning
            warnings.filterwarnings("ignore", category=TracerWarning)
        except ImportError:
            pass

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

    tok_result = _convert_tokenizer_to_openvino(ov_model_path, model_id)
    result.update(tok_result)

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
            _ensure_sdxl_tokenizer_files(ov_model_path, model_id)
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            return {
                "model_path": str(ov_model_path),
                "unet_path": str(unet_path),
                "vae_decoder_path": str(vae_path),
            }

    logger.info("Downloading and exporting SDXL model (this may take a long time)...")
    logger.info("Large files (6+ GB) - download will resume if interrupted.")

    def do_export():
        import torch

        return OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            export=True,
            compile=False,
            load_in_8bit=False,
            torch_dtype=torch.float32,
        )

    pipeline = _download_with_retry(do_export, max_retries=3)

    pipeline.save_pretrained(str(ov_model_path))

    _ensure_sdxl_tokenizer_files(ov_model_path, model_id)

    logger.info(f"OpenVINO model saved to {ov_model_path}")

    return {
        "model_path": str(ov_model_path),
        "unet_path": str(ov_model_path / "unet" / "openvino_model.xml"),
        "vae_decoder_path": str(ov_model_path / "vae_decoder" / "openvino_model.xml"),
        "text_encoder_path": str(ov_model_path / "text_encoder" / "openvino_model.xml"),
        "text_encoder_2_path": str(ov_model_path / "text_encoder_2" / "openvino_model.xml"),
    }


def _ensure_sdxl_tokenizer_files(ov_model_path: Path, model_id: str) -> None:
    """Ensure tokenizer files exist for GenAI Text2ImagePipeline compatibility.

    Saves both HuggingFace-format files (vocab.json, merges.txt, etc.)
    and OpenVINO IR tokenizer models (openvino_tokenizer.xml) for
    tokenizer/ and tokenizer_2/ directories.
    """
    tokenizer_dir = ov_model_path / "tokenizer"
    tokenizer_2_dir = ov_model_path / "tokenizer_2"
    scheduler_cfg = ov_model_path / "scheduler" / "scheduler_config.json"

    # Check HuggingFace tokenizer files
    tok1_hf_ok = (tokenizer_dir / "vocab.json").exists()
    tok2_hf_ok = (tokenizer_2_dir / "vocab.json").exists()
    sched_ok = scheduler_cfg.exists()

    # Check OpenVINO tokenizer IR models
    tok1_ov_ok = (tokenizer_dir / "openvino_tokenizer.xml").exists()
    tok2_ov_ok = (tokenizer_2_dir / "openvino_tokenizer.xml").exists()

    if tok1_hf_ok and tok2_hf_ok and sched_ok and tok1_ov_ok and tok2_ov_ok:
        return

    try:
        from transformers import CLIPTokenizer
    except ImportError:
        logger.warning("transformers not installed, cannot save tokenizer files")
        return

    # --- Save HuggingFace tokenizer files if missing ---

    if not tok1_hf_ok:
        try:
            tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(str(tokenizer_dir))
            logger.info(f"Saved tokenizer to {tokenizer_dir}")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    if not tok2_hf_ok:
        try:
            tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
            tokenizer_2_dir.mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(str(tokenizer_2_dir))
            logger.info(f"Saved tokenizer_2 to {tokenizer_2_dir}")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer_2: {e}")

    if not sched_ok:
        try:
            from diffusers import EulerDiscreteScheduler
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
            scheduler_dir = ov_model_path / "scheduler"
            scheduler_dir.mkdir(parents=True, exist_ok=True)
            scheduler.save_pretrained(str(scheduler_dir))
            logger.info(f"Saved scheduler config to {scheduler_dir}")
        except Exception as e:
            logger.warning(f"Failed to save scheduler config: {e}")

    # --- Convert tokenizers to OpenVINO IR format ---

    if tok1_ov_ok and tok2_ov_ok:
        return

    try:
        import openvino as ov
        from openvino_tokenizers import convert_tokenizer
    except ImportError:
        logger.warning(
            "openvino-tokenizers not installed, skipping OpenVINO tokenizer "
            "conversion. Install with: pip install openvino-tokenizers"
        )
        return

    for subfolder, out_dir, already_ok in [
        ("tokenizer", tokenizer_dir, tok1_ov_ok),
        ("tokenizer_2", tokenizer_2_dir, tok2_ov_ok),
    ]:
        if already_ok:
            continue
        try:
            # convert_tokenizer requires a fast tokenizer.
            # CLIP models only ship a slow CLIPTokenizer (no tokenizer.json),
            # so AutoTokenizer(use_fast=True) still returns the slow version.
            # Convert slow -> fast explicitly via convert_slow_tokenizer.
            from transformers import PreTrainedTokenizerFast
            from transformers.convert_slow_tokenizer import convert_slow_tokenizer

            slow_tok = CLIPTokenizer.from_pretrained(
                model_id, subfolder=subfolder,
            )
            fast_backend = convert_slow_tokenizer(slow_tok)
            fast_tok = PreTrainedTokenizerFast(
                tokenizer_object=fast_backend,
            )

            # Some older openvino_tokenizers C++ extensions don't support
            # the number of BPE node inputs produced by newer Python converter
            # when handle_special_tokens_with_re is enabled. Try conversion
            # with it disabled first (fewer inputs), then fall back to default.
            import inspect
            convert_sig = inspect.signature(convert_tokenizer)
            convert_attempts = []
            if "handle_special_tokens_with_re" in convert_sig.parameters:
                convert_attempts.append(
                    {"with_detokenizer": True, "handle_special_tokens_with_re": False}
                )
            convert_attempts.append({"with_detokenizer": True})

            last_err = None
            for kwargs in convert_attempts:
                try:
                    ov_tokenizer, ov_detokenizer = convert_tokenizer(
                        fast_tok, **kwargs,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e

            if last_err is not None:
                raise last_err

            out_dir.mkdir(parents=True, exist_ok=True)
            ov.save_model(ov_tokenizer, str(out_dir / "openvino_tokenizer.xml"))
            ov.save_model(
                ov_detokenizer, str(out_dir / "openvino_detokenizer.xml"),
            )
            logger.info(f"Saved OpenVINO tokenizer IR to {out_dir}")
        except Exception as e:
            logger.warning(f"Failed to convert {subfolder} to OpenVINO IR: {e}")
            # Fallback: save tokenizer.json so that OpenVINO GenAI can load
            # the tokenizer via the HuggingFace tokenizers library directly.
            try:
                tok_json = out_dir / "tokenizer.json"
                if not tok_json.exists():
                    from transformers.convert_slow_tokenizer import (
                        convert_slow_tokenizer as _convert,
                    )
                    _slow = CLIPTokenizer.from_pretrained(
                        model_id, subfolder=subfolder,
                    )
                    _fast_backend = _convert(_slow)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _fast_backend.save(str(tok_json))
                    logger.info(
                        f"Saved tokenizer.json fallback to {out_dir} "
                        "(update openvino-tokenizers for native OV IR support)"
                    )
            except Exception as e2:
                logger.warning(f"Fallback tokenizer.json save also failed: {e2}")


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


def download_llama_model(
    output_dir: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    export_to_openvino: bool = True,
    weight_format: str = "int8",
    hf_token: Optional[str] = None,
) -> Dict[str, str]:
    """Download and export Llama model to OpenVINO IR format.

    Uses optimum-intel OVModelForCausalLM to export the model from
    HuggingFace to OpenVINO IR, with optional weight compression via NNCF.

    Args:
        output_dir: Directory to save the exported model.
        model_id: HuggingFace model ID.
        export_to_openvino: If True, export to OpenVINO IR (recommended).
        weight_format: Weight format for compression: "fp32", "fp16", "int8", "int4".
        hf_token: HuggingFace access token (for gated models like Meta-Llama).

    Returns:
        Dict with model_path and metadata.
    """
    token = _resolve_hf_token(hf_token)
    if not token:
        logger.warning(
            "No HuggingFace token found. Meta-Llama models are gated and require authentication. "
            "Set HF_TOKEN env var or run: huggingface-cli login"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = model_id.split("/")[-1]

    if export_to_openvino:
        return _export_llama_to_openvino(output_dir, model_id, weight_format, token=token)
    else:
        return _download_llama_from_hf(output_dir, model_id, token=token)


def _download_llama_from_hf(
    output_dir: str, model_id: str, token: Optional[str] = None
) -> Dict[str, str]:
    """Download Llama model from HuggingFace (safetensors/PyTorch format)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for Llama download. "
            "Install with: pip install transformers"
        )

    _configure_hf_download()

    logger.info(f"Downloading Llama model: {model_id}")

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    model_path = output_path / model_name

    if model_path.exists():
        config_file = model_path / "config.json"
        if config_file.exists():
            logger.info(f"Model already exists at {model_path}")
            return {"model_path": str(model_path)}

    def do_download():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            use_safetensors=True,
            token=token,
        )

    model = _download_with_retry(do_download, max_retries=3)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    logger.info(f"Model saved to {model_path}")
    return {"model_path": str(model_path)}


def _export_llama_to_openvino(
    output_dir: str,
    model_id: str,
    weight_format: str = "int8",
    token: Optional[str] = None,
) -> Dict[str, str]:
    """Export Llama model to OpenVINO IR using optimum-intel.

    Uses OVModelForCausalLM.from_pretrained(export=True) for conversion,
    with NNCF weight compression for INT8/INT4 formats.
    """
    try:
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "optimum-intel is required for Llama export. "
            "Install with: pip install optimum[openvino] nncf"
        )

    _configure_hf_download()

    output_path = Path(output_dir)
    model_name = model_id.split("/")[-1]
    ov_model_path = output_path / f"{model_name}-openvino-{weight_format}"

    if ov_model_path.exists():
        config_file = ov_model_path / "config.json"
        xml_files = list(ov_model_path.glob("*.xml"))
        if config_file.exists() and xml_files:
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            result = {"model_path": str(ov_model_path)}
            tok_result = _convert_tokenizer_to_openvino(
                ov_model_path, model_id, token=token,
            )
            result.update(tok_result)
            return result

    logger.info(f"Exporting {model_id} to OpenVINO IR (weight_format={weight_format})...")
    logger.info("This may take a long time for large models...")

    ov_export_kwargs: Dict[str, Any] = {
        "export": True,
        "compile": False,
        "token": token,
    }

    if weight_format in ("int8", "int4"):
        try:
            from optimum.intel import OVWeightQuantizationConfig

            ov_export_kwargs["quantization_config"] = OVWeightQuantizationConfig(
                bits=8 if weight_format == "int8" else 4,
                sym=True if weight_format == "int4" else False,
            )
            logger.info(f"Using NNCF weight-only compression: {weight_format}")
        except ImportError:
            logger.warning(
                "NNCF not available for weight compression. "
                "Exporting with FP16 weights instead. "
                "Install with: pip install nncf"
            )
            weight_format = "fp16"

    def do_export():
        return OVModelForCausalLM.from_pretrained(
            model_id,
            **ov_export_kwargs,
        )

    model = _download_with_retry(do_export, max_retries=3)
    model.save_pretrained(str(ov_model_path))

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.save_pretrained(str(ov_model_path))

    logger.info(f"OpenVINO model saved to {ov_model_path}")

    result = {"model_path": str(ov_model_path)}
    tok_result = _convert_tokenizer_to_openvino(
        ov_model_path, model_id, token=token,
    )
    result.update(tok_result)
    return result


def download_llama2_70b_model(
    output_dir: str,
    model_id: str = "meta-llama/Llama-2-70b-chat-hf",
    export_to_openvino: bool = True,
    weight_format: str = "int4",
    hf_token: Optional[str] = None,
) -> Dict[str, str]:
    """Download and export Llama 2 70B model to OpenVINO IR format.

    Uses the same export pipeline as Llama 3.1 8B but with Llama 2 70B defaults.
    INT4 weight compression is recommended for 70B models to fit in memory.

    Args:
        output_dir: Directory to save the exported model.
        model_id: HuggingFace model ID.
        export_to_openvino: If True, export to OpenVINO IR (recommended).
        weight_format: Weight format: "fp32", "fp16", "int8", "int4" (default: "int4").
        hf_token: HuggingFace access token (for gated models like Meta-Llama).

    Returns:
        Dict with model_path and metadata.
    """
    token = _resolve_hf_token(hf_token)
    if not token:
        logger.warning(
            "No HuggingFace token found. Meta-Llama models are gated and require authentication. "
            "Set HF_TOKEN env var or run: huggingface-cli login"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if export_to_openvino:
        return _export_llama_to_openvino(output_dir, model_id, weight_format, token=token)
    else:
        return _download_llama_from_hf(output_dir, model_id, token=token)


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
