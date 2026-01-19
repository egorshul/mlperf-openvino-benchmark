#!/usr/bin/env python3
"""
Convert MLPerf RetinaNet PyTorch checkpoint to ONNX with dynamic batch support.

This script downloads the official MLPerf RetinaNet checkpoint and converts it
to ONNX format with dynamic batch dimension for efficient batch inference.

The exported model outputs RAW predictions (cls_logits, bbox_regression) without
NMS, allowing efficient batch processing. NMS is done in postprocessing.

Usage:
    python convert_retinanet_to_onnx.py --output models/retinanet_dynamic.onnx
    python convert_retinanet_to_onnx.py --checkpoint /path/to/resnext50_32x4d_fpn.pth --output model.onnx
    python convert_retinanet_to_onnx.py --batch-size 8 --output model_batch8.onnx  # Fixed batch
"""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLPerf official model URLs
PYTORCH_CHECKPOINT_URL = "https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth"
ONNX_MODEL_URL = "https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx"


def download_checkpoint(output_path: str) -> str:
    """Download MLPerf RetinaNet PyTorch checkpoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Checkpoint already exists: {output_path}")
        return str(output_path)

    logger.info(f"Downloading checkpoint from {PYTORCH_CHECKPOINT_URL}")
    logger.info("This may take a few minutes (~160MB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(PYTORCH_CHECKPOINT_URL, str(output_path), progress_hook)
        print()  # New line after progress
        logger.info(f"Downloaded to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise


def build_retinanet_model(num_classes: int = 264):
    """
    Build RetinaNet model with ResNeXt50_32x4d backbone.

    This matches the MLPerf reference model architecture.

    Args:
        num_classes: Number of classes (264 for OpenImages MLPerf)
    """
    try:
        import torch
        import torchvision
        from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Install with: pip install torch torchvision")
        sys.exit(1)

    logger.info("Building RetinaNet with ResNeXt50_32x4d backbone...")

    # Build backbone: ResNeXt50_32x4d with FPN + P6P7
    # IMPORTANT: MLPerf uses returned_layers=[2, 3, 4] which corresponds to C3, C4, C5
    # with 512, 1024, 2048 channels respectively. This matches the checkpoint architecture.
    backbone = resnet_fpn_backbone(
        backbone_name='resnext50_32x4d',
        weights=None,
        trainable_layers=3,  # Default for MLPerf
        returned_layers=[2, 3, 4],  # C3, C4, C5 - matches MLPerf checkpoint
        extra_blocks=LastLevelP6P7(256, 256),
    )

    # Anchor generator matching MLPerf config
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # Build RetinaNet
    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        topk_candidates=1000,
    )

    return model


def load_checkpoint(model, checkpoint_path: str):
    """Load weights from MLPerf checkpoint."""
    import torch

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, torch.jit.ScriptModule):
        # TorchScript model - extract state_dict
        logger.info("Checkpoint is a TorchScript module, extracting state_dict...")
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, torch.nn.Module):
        # Regular PyTorch model
        logger.info("Checkpoint is a PyTorch module, extracting state_dict...")
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # Try to use as state_dict directly
        state_dict = checkpoint

    # Try to load with strict=False first to see mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
        for k in missing[:10]:
            logger.warning(f"  - {k}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            logger.warning(f"  - {k}")

    if not missing and not unexpected:
        logger.info("Checkpoint loaded successfully (all keys matched)")
    else:
        logger.info("Checkpoint loaded with some key mismatches (may still work)")

    return model


import torch

class RetinaNetRawOutputWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that outputs raw predictions without NMS.

    Outputs:
    - boxes: [B, num_anchors, 4] - raw bbox regression
    - scores: [B, num_anchors, num_classes] - raw classification logits

    This allows batch inference; NMS is done in postprocessing.
    """
    def __init__(self, model, num_classes: int = 264):
        import torch
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.num_classes = num_classes

    def forward(self, images):
        import torch

        # Get FPN features
        features = self.backbone(images)
        features_list = list(features.values())

        # Get raw predictions from head
        head_outputs = self.head(features_list)

        cls_logits_list = head_outputs['cls_logits']  # List of [B, A*C, H, W]
        bbox_regression_list = head_outputs['bbox_regression']  # List of [B, A*4, H, W]

        batch_size = images.shape[0]

        all_cls = []
        all_bbox = []

        for cls, bbox in zip(cls_logits_list, bbox_regression_list):
            B, _, H, W = cls.shape
            # Reshape: [B, A*C, H, W] -> [B, H*W*A, C]
            num_anchors_per_loc = cls.shape[1] // self.num_classes
            cls = cls.view(B, num_anchors_per_loc, self.num_classes, H, W)
            cls = cls.permute(0, 3, 4, 1, 2).reshape(B, -1, self.num_classes)

            # Reshape bbox: [B, A*4, H, W] -> [B, H*W*A, 4]
            bbox = bbox.view(B, num_anchors_per_loc, 4, H, W)
            bbox = bbox.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)

            all_cls.append(cls)
            all_bbox.append(bbox)

        # Concatenate all FPN levels
        cls_logits = torch.cat(all_cls, dim=1)  # [B, total_anchors, num_classes]
        bbox_regression = torch.cat(all_bbox, dim=1)  # [B, total_anchors, 4]

        return bbox_regression, cls_logits


class RetinaNetWithNMSWrapper(torch.nn.Module):
    """
    Wrapper that includes NMS in the model (batch_size=1 only).

    Outputs format matches official MLPerf ONNX model:
    - boxes: [num_detections, 4]
    - scores: [num_detections]
    - labels: [num_detections]
    """
    def __init__(self, model):
        import torch
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, images):
        # Use torchvision's built-in postprocessing with NMS
        with torch.no_grad():
            outputs = self.model(images)

        # outputs is list of dicts, take first (batch=1)
        out = outputs[0]
        return out['boxes'], out['scores'], out['labels']


def export_to_onnx(
    model,
    output_path: str,
    batch_size: int = None,
    image_size: int = 800,
    opset: int = 17,
    include_nms: bool = False,
    num_classes: int = 264,
):
    """
    Export RetinaNet to ONNX format.

    Args:
        model: PyTorch RetinaNet model
        output_path: Output ONNX file path
        batch_size: Fixed batch size (None = dynamic)
        image_size: Input image size (default 800 for MLPerf)
        opset: ONNX opset version
        include_nms: If True, include NMS (batch=1 only). If False, output raw predictions.
        num_classes: Number of classes
    """
    import torch

    model.eval()

    if include_nms:
        if batch_size is not None and batch_size > 1:
            logger.warning("NMS mode only supports batch_size=1, ignoring batch_size setting")
        wrapper = RetinaNetWithNMSWrapper(model)
        dummy_batch = 1
        input_names = ['images']
        output_names = ['boxes', 'scores', 'labels']
        dynamic_axes = None  # NMS output shapes are dynamic by nature
        logger.info("Exporting WITH NMS (batch_size=1 only)")
    else:
        wrapper = RetinaNetRawOutputWrapper(model, num_classes=num_classes)
        dummy_batch = batch_size if batch_size else 1
        input_names = ['images']
        output_names = ['bbox_regression', 'cls_logits']

        if batch_size is None:
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'bbox_regression': {0: 'batch_size'},
                'cls_logits': {0: 'batch_size'},
            }
            logger.info("Exporting WITHOUT NMS, with dynamic batch size")
        else:
            dynamic_axes = None
            logger.info(f"Exporting WITHOUT NMS, with fixed batch size: {batch_size}")

    # Create dummy input
    dummy_input = torch.randn(dummy_batch, 3, image_size, image_size)

    logger.info(f"Input shape: [{dummy_batch}, 3, {image_size}, {image_size}]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    logger.info(f"Exporting to ONNX (opset {opset})...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Exported to {output_path}")

    # Verify
    verify_onnx(str(output_path), batch_size, image_size, include_nms)

    return str(output_path)


def verify_onnx(onnx_path: str, batch_size: int = None, image_size: int = 800, include_nms: bool = False):
    """Verify the exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnx/onnxruntime not installed, skipping verification")
        return

    logger.info("Verifying ONNX model...")

    # Load and check
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    logger.info("ONNX model structure check passed")

    # Print IO info
    logger.info("Model inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        logger.info(f"  {inp.name}: {shape}")

    logger.info("Model outputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        logger.info(f"  {out.name}: {shape}")

    # Test inference
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    if include_nms:
        test_batches = [1]
    else:
        test_batches = [1, 2, 4] if batch_size is None else [batch_size]

    logger.info("Testing inference:")
    for bs in test_batches:
        dummy = np.random.randn(bs, 3, image_size, image_size).astype(np.float32)
        try:
            outputs = sess.run(None, {'images': dummy})
            shapes = [f"{o.shape}" for o in outputs]
            logger.info(f"  Batch {bs}: OK - output shapes: {shapes}")
        except Exception as e:
            logger.error(f"  Batch {bs}: FAILED - {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MLPerf RetinaNet to ONNX with dynamic batch support'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to PyTorch checkpoint (will download if not specified)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models/retinanet_dynamic_batch.onnx',
        help='Output ONNX file path'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Fixed batch size (default: dynamic)'
    )
    parser.add_argument(
        '--image-size', '-s',
        type=int,
        default=800,
        help='Input image size (default: 800)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (default: 17)'
    )
    parser.add_argument(
        '--with-nms',
        action='store_true',
        help='Include NMS in model (batch_size=1 only, like official MLPerf model)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=264,
        help='Number of classes (default: 264 for OpenImages)'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download checkpoint, do not convert'
    )

    args = parser.parse_args()

    # Import torch here to check availability
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch is required. Install with: pip install torch torchvision")
        sys.exit(1)

    # Download checkpoint if needed
    if args.checkpoint is None:
        checkpoint_dir = Path(args.output).parent / 'checkpoints'
        checkpoint_path = checkpoint_dir / 'resnext50_32x4d_fpn.pth'
        args.checkpoint = download_checkpoint(str(checkpoint_path))

    if args.download_only:
        logger.info("Download complete. Exiting.")
        return

    # Build and load model
    model = build_retinanet_model(num_classes=args.num_classes)
    model = load_checkpoint(model, args.checkpoint)

    # Export to ONNX
    export_to_onnx(
        model,
        args.output,
        batch_size=args.batch_size,
        image_size=args.image_size,
        opset=args.opset,
        include_nms=args.with_nms,
        num_classes=args.num_classes,
    )

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)

    if args.with_nms:
        logger.info("\nModel exported WITH NMS (batch_size=1 only)")
        logger.info("Output format: boxes, scores, labels (post-NMS)")
    else:
        logger.info("\nModel exported WITHOUT NMS (supports dynamic batch)")
        logger.info("Output format: bbox_regression, cls_logits (raw)")
        logger.info("NMS should be done in postprocessing (already implemented in retinanet_sut.py)")

    logger.info(f"\nTo use the model:")
    logger.info(f"  mlperf-ov run --model retinanet --model-path {args.output}")


if __name__ == '__main__':
    main()
