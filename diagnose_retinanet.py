#!/usr/bin/env python3
"""
Diagnostic script to debug RetinaNet through the benchmark runner.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def main():
    from mlperf_openvino.core.config import BenchmarkConfig, TestMode
    from mlperf_openvino.core.benchmark_runner import BenchmarkRunner

    print("=" * 60)
    print("RetinaNet Diagnostic")
    print("=" * 60)

    # Download model if needed
    from mlperf_openvino.utils.model_downloader import download_model
    model_path = download_model("retinanet", "models")
    print(f"\nModel path: {model_path}")

    # Create config
    config = BenchmarkConfig.default_retinanet()
    config.model.model_path = model_path
    config.dataset.path = "./data/openimages"
    config.dataset.num_samples = 10  # Small test
    config.test_mode = TestMode.ACCURACY_ONLY

    print(f"\nConfig:")
    print(f"  Model path: {config.model.model_path}")
    print(f"  Data path: {config.dataset.path}")
    print(f"  Num samples: {config.dataset.num_samples}")

    # Check annotations file
    data_path = Path(config.dataset.path)
    ann_paths = [
        data_path / "annotations" / "openimages-mlperf.json",
        data_path / "openimages-mlperf.json",
    ]

    ann_file = None
    for p in ann_paths:
        if p.exists():
            ann_file = p
            break

    print(f"\nAnnotations file: {ann_file}")
    if ann_file:
        import json
        with open(ann_file) as f:
            coco = json.load(f)
        num_images = len(coco.get('images', []))
        num_anns = len(coco.get('annotations', []))
        cats = coco.get('categories', [])
        cat_ids = [c['id'] for c in cats]
        print(f"  Images in annotations: {num_images}")
        print(f"  Annotations: {num_anns}")
        print(f"  Categories: {len(cats)}, range: {min(cat_ids)}-{max(cat_ids)}")

        # Check if num_samples matches
        if num_images < config.dataset.num_samples:
            print(f"\n  WARNING: Annotations have {num_images} images but config wants {config.dataset.num_samples}!")
            print(f"  Setting num_samples to {num_images}")
            config.dataset.num_samples = num_images
    else:
        print("  NOT FOUND!")
        return

    # Create runner
    print("\nCreating runner...")
    runner = BenchmarkRunner(config)
    runner.setup()

    print(f"\nSUT type: {type(runner.sut).__name__}")
    print(f"QSL count: {runner.qsl.total_sample_count}")

    # Run small test
    print("\nRunning inference...")
    results = runner.run()

    print(f"\nResults from runner.run(): {results}")

    # Check predictions
    predictions = runner.sut.get_predictions()
    print(f"\nPredictions count: {len(predictions)}")

    if predictions:
        # Analyze first prediction
        first_idx = list(predictions.keys())[0]
        pred = predictions[first_idx]
        print(f"\nFirst prediction (sample {first_idx}):")
        print(f"  Boxes: {pred['boxes'].shape if len(pred['boxes']) > 0 else 'empty'}")
        print(f"  Scores: {pred['scores'].shape if len(pred['scores']) > 0 else 'empty'}")
        print(f"  Labels: {pred['labels'].shape if len(pred['labels']) > 0 else 'empty'}")

        if len(pred['labels']) > 0:
            print(f"  Label values (first 10): {pred['labels'][:10]}")
            print(f"  Label range: {pred['labels'].min()} to {pred['labels'].max()}")
            print(f"  Expected range for COCO: 1-365 (after +1 offset)")
            print(f"  Model outputs 0-indexed: 0-364")

        # Manually test evaluation
        print("\n" + "=" * 60)
        print("Manual evaluation test:")
        print("=" * 60)

        from mlperf_openvino.datasets.coco_eval import evaluate_openimages_accuracy

        result = evaluate_openimages_accuracy(
            predictions=predictions,
            coco_annotations_file=str(ann_file),
            input_size=800,
            model_labels_zero_indexed=True,
            boxes_in_pixels=True,
        )
        print(f"\nManual evaluate_openimages_accuracy result: {result}")
    else:
        print("\nNO PREDICTIONS!")

if __name__ == "__main__":
    main()
