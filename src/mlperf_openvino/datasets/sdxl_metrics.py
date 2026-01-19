"""
SDXL accuracy metrics: FID and CLIP score computation.

MLPerf SDXL accuracy targets:
- FID: [23.01085758, 23.95007626]
- CLIP: [31.68631873, 31.81331801]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# MLPerf accuracy targets
FID_TARGET_MIN = 23.01085758
FID_TARGET_MAX = 23.95007626
CLIP_TARGET_MIN = 31.68631873
CLIP_TARGET_MAX = 31.81331801


def compute_fid_clip_scores(
    generated_images: List[np.ndarray],
    prompts: List[str],
    data_path: str,
    reference_images_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute FID and CLIP scores for generated images.

    Args:
        generated_images: List of generated images [H, W, C]
        prompts: List of text prompts
        data_path: Path to dataset directory
        reference_images_path: Path to reference images (for FID)

    Returns:
        Dictionary with fid, clip_score, and validation status
    """
    results = {
        "fid": 0.0,
        "clip_score": 0.0,
        "num_samples": len(generated_images),
        "fid_valid": False,
        "clip_valid": False,
    }

    if len(generated_images) == 0:
        return results

    # Compute CLIP score
    try:
        clip_score = compute_clip_score(generated_images, prompts)
        results["clip_score"] = clip_score
        results["clip_valid"] = CLIP_TARGET_MIN <= clip_score <= CLIP_TARGET_MAX
    except Exception as e:
        logger.error(f"Failed to compute CLIP score: {e}")

    # Compute FID score
    try:
        data_path = Path(data_path)
        ref_path = reference_images_path or data_path / "images"

        if Path(ref_path).exists():
            fid = compute_fid(generated_images, str(ref_path))
            results["fid"] = fid
            results["fid_valid"] = FID_TARGET_MIN <= fid <= FID_TARGET_MAX
        else:
            logger.warning(f"Reference images not found at {ref_path}")
    except Exception as e:
        logger.error(f"Failed to compute FID: {e}")

    return results


def compute_clip_score(
    images: List[np.ndarray],
    prompts: List[str],
) -> float:
    """
    Compute CLIP score between images and prompts.

    The CLIP score measures how well the generated images match
    the input text prompts.

    Args:
        images: List of generated images [H, W, C]
        prompts: List of text prompts

    Returns:
        Average CLIP score
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError(
            "transformers and torch are required for CLIP score. "
            "Install with: pip install transformers torch"
        )

    logger.info("Computing CLIP score...")

    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    scores = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            # Process inputs
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)

            # Compute similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity (diagonal for matching pairs)
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            scores.extend(similarity.cpu().numpy().tolist())

    # CLIP score is typically scaled to [0, 100]
    avg_score = np.mean(scores) * 100

    logger.info(f"CLIP score: {avg_score:.4f}")
    return avg_score


def compute_fid(
    generated_images: List[np.ndarray],
    reference_path: str,
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between generated
    and reference images.

    Lower FID indicates better quality and diversity.

    Args:
        generated_images: List of generated images [H, W, C]
        reference_path: Path to reference images directory

    Returns:
        FID score
    """
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import inception_v3
        from scipy import linalg
    except ImportError:
        raise ImportError(
            "torch, torchvision, and scipy are required for FID. "
            "Install with: pip install torch torchvision scipy"
        )

    logger.info("Computing FID score...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Inception model
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove final layer
    inception = inception.to(device)
    inception.eval()

    # Image transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_activations(images: List[np.ndarray]) -> np.ndarray:
        """Extract Inception features from images."""
        features = []
        batch_size = 8

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                tensors = torch.stack([transform(img) for img in batch])
                tensors = tensors.to(device)

                acts = inception(tensors)
                features.append(acts.cpu().numpy())

        return np.concatenate(features, axis=0)

    def load_reference_images(path: str, max_count: int = 5000) -> List[np.ndarray]:
        """Load reference images from directory."""
        from PIL import Image

        ref_path = Path(path)
        images = []

        extensions = [".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"]
        files = []
        for ext in extensions:
            files.extend(ref_path.glob(f"*{ext}"))

        files = sorted(files)[:max_count]

        for f in files:
            img = Image.open(f).convert("RGB")
            img = np.array(img)
            images.append(img)

        return images

    # Get activations for generated images
    gen_acts = get_activations(generated_images)

    # Get activations for reference images
    ref_images = load_reference_images(reference_path, len(generated_images))
    if len(ref_images) == 0:
        logger.warning("No reference images found")
        return 0.0

    ref_acts = get_activations(ref_images)

    # Compute FID
    mu1, sigma1 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    mu2, sigma2 = ref_acts.mean(axis=0), np.cov(ref_acts, rowvar=False)

    # Compute squared difference of means
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # Compute sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical instabilities
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    logger.info(f"FID score: {fid:.4f}")
    return float(fid)


def validate_accuracy(fid: float, clip_score: float) -> Dict[str, bool]:
    """
    Check if accuracy metrics meet MLPerf targets.

    Args:
        fid: FID score
        clip_score: CLIP score

    Returns:
        Dictionary with validation results
    """
    return {
        "fid_valid": FID_TARGET_MIN <= fid <= FID_TARGET_MAX,
        "clip_valid": CLIP_TARGET_MIN <= clip_score <= CLIP_TARGET_MAX,
        "overall_valid": (
            FID_TARGET_MIN <= fid <= FID_TARGET_MAX and
            CLIP_TARGET_MIN <= clip_score <= CLIP_TARGET_MAX
        ),
    }
