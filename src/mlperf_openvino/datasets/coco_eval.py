"""
COCO evaluation for RetinaNet Object Detection.

Uses pycocotools for accurate mAP calculation following MLPerf reference implementation.
Based on: https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/accuracy-openimages.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not available. Install with: pip install pycocotools")


def evaluate_openimages_accuracy(
    predictions: Dict[int, Dict[str, np.ndarray]],
    coco_annotations_file: str,
    image_sizes: Optional[Dict[int, Tuple[int, int]]] = None,
    input_size: int = 800,
    model_labels_zero_indexed: bool = True,
    boxes_in_pixels: bool = True,
) -> Dict[str, float]:
    """
    Evaluate RetinaNet predictions using COCO API (official MLPerf method).

    Args:
        predictions: Dict of {sample_idx: {'boxes': [N,4], 'scores': [N], 'labels': [N]}}
        coco_annotations_file: Path to COCO format annotations JSON
        image_sizes: Dict of {sample_idx: (width, height)} - used if boxes need scaling
        input_size: Model input size (default 800 for RetinaNet)
        model_labels_zero_indexed: If True, add +1 to labels. MLPerf ONNX model outputs
                                   0-indexed labels (0-364), COCO uses 1-indexed category_ids (1-365).
        boxes_in_pixels: If True, boxes are already in pixel coordinates [0, input_size].
                        MLPerf ONNX RetinaNet outputs pixels, so this should be True.

    Returns:
        Dictionary with mAP metrics
    """
    if not PYCOCOTOOLS_AVAILABLE:
        logger.error("pycocotools not available, cannot compute accurate mAP")
        return {'mAP': 0.0, 'error': 'pycocotools not installed'}

    # Load COCO ground truth
    logger.info(f"Loading COCO annotations from {coco_annotations_file}")
    coco_gt = COCO(coco_annotations_file)

    # Build image_id to sample_idx mapping
    # COCO image IDs are typically 1-indexed
    image_id_to_sample = {}
    sample_to_image_id = {}

    # Get sorted image IDs to create proper mapping
    sorted_img_ids = sorted(coco_gt.imgs.keys())
    for sample_idx, img_id in enumerate(sorted_img_ids):
        image_id_to_sample[img_id] = sample_idx
        sample_to_image_id[sample_idx] = img_id

    logger.info(f"COCO has {len(coco_gt.imgs)} images, {len(coco_gt.cats)} categories")

    # Log available categories for debugging
    cat_ids = sorted(coco_gt.cats.keys())
    logger.debug(f"Category IDs: {cat_ids[:10]}... (total {len(cat_ids)})")

    # Check if annotations have correct category_id range (1-365 for MLPerf)
    # If we see category_ids > 365, the annotations file needs regeneration
    max_cat_id = max(cat_ids) if cat_ids else 0
    min_cat_id = min(cat_ids) if cat_ids else 0
    if max_cat_id > 400:  # Old format used all ~600 OpenImages classes
        logger.warning(f"COCO annotations have category_ids up to {max_cat_id}")
        logger.warning("This suggests the annotations file was created with OLD format.")
        logger.warning("Please delete openimages-mlperf.json and re-run to regenerate with correct category_ids (1-365).")

    # Auto-detect label format by analyzing first predictions
    # Check what label values the model outputs
    all_model_labels = []
    for pred in list(predictions.values())[:100]:  # Sample first 100 predictions
        labels = pred.get('labels', np.array([]))
        if len(labels) > 0:
            all_model_labels.extend(labels[:10].tolist())  # First 10 labels per image

    if all_model_labels:
        min_label = int(min(all_model_labels))
        max_label = int(max(all_model_labels))
        logger.info(f"Model label range: {min_label} to {max_label}")
        logger.info(f"COCO category_id range: {min_cat_id} to {max_cat_id}")

        # Auto-detect if we need to adjust label indexing
        if model_labels_zero_indexed:
            # If model outputs 0-indexed and min is 0, we add +1
            if min_label == 0 and min_cat_id == 1:
                logger.info("Auto-detected: Model uses 0-indexed labels, will add +1")
            elif min_label >= 1 and min_cat_id == 1:
                logger.warning(f"Model labels start at {min_label}, but model_labels_zero_indexed=True")
                logger.warning("Labels may already be 1-indexed. Consider setting model_labels_zero_indexed=False")

    # Convert predictions to COCO format
    # COCO format: [{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}]
    coco_results = []
    skipped_categories = set()
    valid_cat_ids = set(coco_gt.cats.keys())

    # Debug: log first few predictions
    debug_count = 0

    for sample_idx, pred in predictions.items():
        boxes = pred.get('boxes', np.array([]))
        scores = pred.get('scores', np.array([]))
        labels = pred.get('labels', np.array([]))

        if len(boxes) == 0:
            continue

        # Get image_id for this sample
        if sample_idx in sample_to_image_id:
            image_id = sample_to_image_id[sample_idx]
        else:
            image_id = sample_idx + 1  # Fallback: assume 1-indexed

        # Get image dimensions for denormalization
        if image_id in coco_gt.imgs:
            img_info = coco_gt.imgs[image_id]
            img_width = img_info.get('width', input_size)
            img_height = img_info.get('height', input_size)
        elif image_sizes and sample_idx in image_sizes:
            img_width, img_height = image_sizes[sample_idx]
        else:
            img_width = img_height = input_size

        for box, score, label in zip(boxes, scores, labels):
            # MLPerf RetinaNet ONNX model outputs boxes in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = box

            if boxes_in_pixels:
                # Boxes already in pixel coordinates [0, input_size]
                # Just use directly
                x1_px, y1_px, x2_px, y2_px = x1, y1, x2, y2
            else:
                # Boxes are normalized [0, 1] - scale to image dimensions
                x1_px = x1 * img_width
                y1_px = y1 * img_height
                x2_px = x2 * img_width
                y2_px = y2 * img_height

            # Convert to COCO format [x, y, w, h]
            bbox_width = x2_px - x1_px
            bbox_height = y2_px - y1_px

            # Category ID - model outputs ORIGINAL OpenImages category IDs
            category_id = int(label)
            if model_labels_zero_indexed:
                category_id = category_id + 1

            # Skip invalid categories
            if category_id not in valid_cat_ids:
                skipped_categories.add(category_id)
                continue

            # Debug output for first few predictions
            if debug_count < 5:
                logger.info(f"Pred {debug_count}: img={image_id}, cat={category_id}, "
                           f"box=[{x1_px:.1f},{y1_px:.1f},{bbox_width:.1f},{bbox_height:.1f}], score={score:.3f}")
                debug_count += 1

            coco_results.append({
                'image_id': int(image_id),
                'category_id': category_id,
                'bbox': [float(x1_px), float(y1_px), float(bbox_width), float(bbox_height)],
                'score': float(score),
            })

    if skipped_categories:
        logger.warning(f"Skipped {len(skipped_categories)} invalid category IDs: {sorted(skipped_categories)[:20]}")

        # If we skipped too many and they look like off-by-one error, suggest fix
        if 0 in skipped_categories and model_labels_zero_indexed:
            logger.warning("Category 0 was skipped - model might output 0-indexed labels starting from 0")
        if len(skipped_categories) > 100:
            skipped_list = sorted(skipped_categories)
            if skipped_list[0] == 0 or (skipped_list[0] > max_cat_id):
                logger.error("Many predictions skipped! Check model_labels_zero_indexed parameter.")
                logger.error(f"Expected category_ids: {min_cat_id}-{max_cat_id}, got labels that map to: {skipped_list[:10]}...")

    if not coco_results:
        logger.error("No predictions to evaluate!")
        logger.error("This indicates a category_id mismatch between model and annotations.")
        if skipped_categories:
            logger.error(f"All {len(skipped_categories)} unique label values were invalid.")
            logger.error(f"Model label values (after +1 if applied): {sorted(skipped_categories)[:20]}")
            logger.error(f"Valid COCO category_ids: {min_cat_id} to {max_cat_id}")

            # Suggest fix
            if all(c > max_cat_id for c in skipped_categories):
                logger.error("FIX: Model labels are too high. Try model_labels_zero_indexed=False")
            elif all(c < min_cat_id for c in skipped_categories):
                logger.error("FIX: Model labels are too low. Try model_labels_zero_indexed=True")
        return {'mAP': 0.0, 'num_predictions': 0}

    logger.info(f"Evaluating {len(coco_results)} predictions")

    # Run COCO evaluation
    try:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        # stats[0] = mAP @ IoU=0.50:0.95
        # stats[1] = mAP @ IoU=0.50
        # stats[2] = mAP @ IoU=0.75
        results = {
            'mAP': float(coco_eval.stats[0]),
            'mAP@0.5': float(coco_eval.stats[1]),
            'mAP@0.75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
            'num_predictions': len(coco_results),
        }

        logger.info(f"mAP@0.5:0.95 = {results['mAP']:.4f}")
        logger.info(f"mAP@0.5 = {results['mAP@0.5']:.4f}")

        return results

    except Exception as e:
        logger.error(f"COCO evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'error': str(e)}


def convert_predictions_for_mlperf(
    predictions: Dict[int, Dict[str, np.ndarray]],
    image_heights: Dict[int, int],
    image_widths: Dict[int, int],
) -> List[np.ndarray]:
    """
    Convert predictions to MLPerf accuracy log format.

    MLPerf format: 7-element array per detection
    [image_id, ymin, xmin, ymax, xmax, score, class]

    Args:
        predictions: Dict of predictions
        image_heights: Dict of image heights
        image_widths: Dict of image widths

    Returns:
        List of prediction arrays in MLPerf format
    """
    results = []

    for sample_idx, pred in predictions.items():
        boxes = pred.get('boxes', np.array([]))
        scores = pred.get('scores', np.array([]))
        labels = pred.get('labels', np.array([]))

        if len(boxes) == 0:
            continue

        for box, score, label in zip(boxes, scores, labels):
            # MLPerf format: [id, ymin, xmin, ymax, xmax, score, class]
            # boxes are in [x1, y1, x2, y2] normalized format
            x1, y1, x2, y2 = box

            detection = np.array([
                sample_idx,  # image_id (or sample index)
                y1,          # ymin (normalized)
                x1,          # xmin (normalized)
                y2,          # ymax (normalized)
                x2,          # xmax (normalized)
                score,
                label,
            ], dtype=np.float32)

            results.append(detection)

    return results
