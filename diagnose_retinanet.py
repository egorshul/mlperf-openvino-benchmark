#!/usr/bin/env python3
"""
Diagnostic script to debug RetinaNet through the benchmark runner.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def main():
    from mlperf_openvino.core.config import BenchmarkConfig, TestMode
    from mlperf_openvino.core.benchmark_runner import BenchmarkRunner

    print("=" * 60)
    print("RetinaNet Diagnostic")
    print("=" * 60)

    # Create config
    config = BenchmarkConfig.default_retinanet()
    config.dataset.path = "./data/openimages"
    config.dataset.num_samples = 10  # Small test
    config.test_mode = TestMode.ACCURACY_ONLY

    print(f"\nConfig:")
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
        print(f"  Images: {len(coco.get('images', []))}")
        print(f"  Annotations: {len(coco.get('annotations', []))}")
        cats = coco.get('categories', [])
        cat_ids = [c['id'] for c in cats]
        print(f"  Categories: {len(cats)}, range: {min(cat_ids)}-{max(cat_ids)}")
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
    print("\nRunning inference on 10 samples...")
    results = runner.run()

    print(f"\nResults: {results}")

    # Check predictions
    predictions = runner.sut.get_predictions()
    print(f"\nPredictions count: {len(predictions)}")

    if predictions:
        # Analyze first prediction
        first_idx = list(predictions.keys())[0]
        pred = predictions[first_idx]
        print(f"\nFirst prediction (sample {first_idx}):")
        print(f"  Boxes shape: {pred['boxes'].shape if len(pred['boxes']) > 0 else 'empty'}")
        print(f"  Scores shape: {pred['scores'].shape if len(pred['scores']) > 0 else 'empty'}")
        print(f"  Labels shape: {pred['labels'].shape if len(pred['labels']) > 0 else 'empty'}")

        if len(pred['labels']) > 0:
            print(f"  Label values: {pred['labels'][:10]}")
            print(f"  Label range: {pred['labels'].min()} to {pred['labels'].max()}")
    else:
        print("\nNO PREDICTIONS!")

if __name__ == "__main__":
    main()
