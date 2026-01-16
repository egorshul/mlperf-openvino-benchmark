#!/usr/bin/env python3
"""
Quick test script for RetinaNet accuracy verification.

Downloads a small subset of OpenImages, runs inference, and evaluates mAP.
Use this to verify the accuracy calculation is working correctly.

Usage:
    python test_retinanet_accuracy.py [--count 50] [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test RetinaNet accuracy")
    parser.add_argument("--count", type=int, default=50, help="Number of images to test")
    parser.add_argument("--force", action="store_true", help="Force re-download dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    args = parser.parse_args()

    print("=" * 60)
    print("RetinaNet Accuracy Test")
    print("=" * 60)

    # Step 1: Download model
    print("\n1. Downloading RetinaNet model...")
    from mlperf_openvino.utils.model_downloader import download_model

    model_result = download_model("retinanet", args.model_dir)
    model_path = model_result.get("onnx_path") or model_result.get("model_path")
    print(f"   Model: {model_path}")

    # Step 2: Download dataset subset
    print(f"\n2. Downloading OpenImages ({args.count} images)...")
    from mlperf_openvino.utils.dataset_downloader import download_openimages

    dataset_result = download_openimages(
        output_dir=args.data_dir,
        max_images=args.count,
        force=args.force,
    )
    data_path = dataset_result["data_path"]
    annotations_file = dataset_result.get("annotations_file")
    print(f"   Data: {data_path}")
    print(f"   Annotations: {annotations_file}")
    print(f"   Images: {dataset_result.get('num_samples', 'unknown')}")

    # Step 3: Verify annotations file
    print("\n3. Verifying COCO annotations...")
    import json

    if annotations_file and Path(annotations_file).exists():
        with open(annotations_file) as f:
            coco = json.load(f)

        num_images = len(coco.get('images', []))
        num_annotations = len(coco.get('annotations', []))
        categories = coco.get('categories', [])
        cat_ids = [c['id'] for c in categories]

        print(f"   Images: {num_images}")
        print(f"   Annotations: {num_annotations}")
        print(f"   Categories: {len(categories)}")
        print(f"   Category ID range: {min(cat_ids)} to {max(cat_ids)}")
        print(f"   First 5 categories: {categories[:5]}")
    else:
        print("   ERROR: Annotations file not found!")
        return 1

    # Step 4: Load model with OpenVINO
    print("\n4. Loading model with OpenVINO...")
    import openvino as ov
    import numpy as np

    core = ov.Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")

    # Map output names to indices
    output_names = {out.any_name: i for i, out in enumerate(model.outputs)}
    print(f"   Outputs: {output_names}")

    # Determine output indices by name patterns
    boxes_idx = scores_idx = labels_idx = None
    for name, idx in output_names.items():
        name_lower = name.lower()
        if 'box' in name_lower or 'bbox' in name_lower:
            boxes_idx = idx
        elif 'score' in name_lower or 'conf' in name_lower:
            scores_idx = idx
        elif 'label' in name_lower or 'class' in name_lower:
            labels_idx = idx

    # Fallback to positional
    if boxes_idx is None:
        boxes_idx = 0
    if labels_idx is None:
        labels_idx = 1 if len(output_names) > 1 else 0
    if scores_idx is None:
        scores_idx = 2 if len(output_names) > 2 else 1

    print(f"   boxes_idx={boxes_idx}, labels_idx={labels_idx}, scores_idx={scores_idx}")

    # Step 5: Load dataset and run inference
    print("\n5. Running inference on test images...")
    from mlperf_openvino.datasets.openimages import OpenImagesDataset

    dataset = OpenImagesDataset(
        data_path=data_path,
        annotations_file=annotations_file,
        count=args.count,
        input_size=800,
        cache_preprocessed=False,  # Don't use cache for test
    )
    dataset.load()

    print(f"   Dataset loaded: {dataset.total_count} images")

    predictions = {}

    for i in range(min(args.count, dataset.total_count)):
        try:
            features = dataset.get_item(i)
            img = features['input']

            # Run inference
            result = compiled(img)

            # Get outputs using determined indices
            boxes = result[boxes_idx]
            scores = result[scores_idx]
            labels = result[labels_idx]

            # Handle batch dimension
            if boxes.ndim == 3:
                boxes = boxes[0]
            if scores.ndim == 2:
                scores = scores[0]
            if labels.ndim == 2:
                labels = labels[0]

            # Filter by score
            mask = scores > 0.05
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            predictions[i] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            }

            if i < 3:
                print(f"   Image {i}: {len(boxes)} detections, labels={labels[:5].astype(int)}")
            elif i == 3:
                print("   ...")

        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            continue

    print(f"   Processed {len(predictions)} images")

    # Step 6: Analyze label distribution
    print("\n6. Analyzing model output labels...")
    all_labels = []
    for pred in predictions.values():
        labels = pred.get('labels', np.array([]))
        if len(labels) > 0:
            all_labels.extend(labels.tolist())

    if all_labels:
        min_label = int(min(all_labels))
        max_label = int(max(all_labels))
        unique_labels = sorted(set(int(l) for l in all_labels))
        print(f"   Total detections: {len(all_labels)}")
        print(f"   Label range: {min_label} to {max_label}")
        print(f"   Unique labels: {len(unique_labels)}")
        print(f"   Sample labels: {unique_labels[:20]}")

        # Check if labels need +1 adjustment
        if min_label == 0:
            print("   Note: Labels start at 0 (0-indexed)")
            print("   Will add +1 to convert to category_ids (1-indexed)")
        elif min_label == 1:
            print("   Note: Labels start at 1 (1-indexed)")
    else:
        print("   ERROR: No detections found!")
        return 1

    # Step 7: Evaluate accuracy
    print("\n7. Evaluating mAP accuracy...")
    from mlperf_openvino.datasets.coco_eval import evaluate_openimages_accuracy

    results = evaluate_openimages_accuracy(
        predictions=predictions,
        coco_annotations_file=annotations_file,
        input_size=800,
        model_labels_zero_indexed=True,  # Model outputs 0-indexed, add +1
        boxes_in_pixels=True,            # Model outputs pixel coordinates
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:0.95 = {results.get('mAP', 0.0):.4f}")
    print(f"mAP@0.5      = {results.get('mAP@0.5', 0.0):.4f}")
    print(f"mAP@0.75     = {results.get('mAP@0.75', 0.0):.4f}")
    print(f"Predictions  = {results.get('num_predictions', 0)}")

    if results.get('mAP', 0.0) > 0:
        print("\nSUCCESS: Accuracy calculation is working!")
    else:
        print("\nWARNING: mAP = 0.0 - check label mapping!")
        if 'error' in results:
            print(f"Error: {results['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
