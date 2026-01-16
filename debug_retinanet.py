#!/usr/bin/env python3
"""
Debug script to understand RetinaNet model output format.
Run this to see exactly what the model outputs.
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("=" * 60)
    print("RetinaNet Debug Script")
    print("=" * 60)

    # 1. Check if model exists
    print("\n1. Checking model...")
    model_paths = [
        Path("models/retinanet.onnx"),
        Path.home() / "models" / "retinanet.onnx",
        Path("data/models/retinanet.onnx"),
    ]

    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = p
            break

    if not model_path:
        print("   Model not found! Downloading...")
        from mlperf_openvino.utils.model_downloader import download_model
        result = download_model("retinanet", "models")
        model_path = Path(result.get("onnx_path") or result.get("model_path"))
        print(f"   Downloaded to: {model_path}")
    else:
        print(f"   Found: {model_path}")

    # 2. Load model with OpenVINO and check outputs
    print("\n2. Loading model with OpenVINO...")
    try:
        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(model_path))

        print(f"   Inputs: {len(model.inputs)}")
        for inp in model.inputs:
            print(f"     - {inp.any_name}: {inp.partial_shape}")

        print(f"   Outputs: {len(model.outputs)}")
        for out in model.outputs:
            print(f"     - {out.any_name}: {out.partial_shape}")

        # Compile and run test
        print("\n3. Running test inference...")
        compiled = core.compile_model(model, "CPU")

        # Create dummy input [1, 3, 800, 800]
        dummy_input = np.random.rand(1, 3, 800, 800).astype(np.float32)
        result = compiled(dummy_input)

        print("\n4. Analyzing outputs:")
        for i, out in enumerate(model.outputs):
            data = result[i]  # Use index instead of output object
            name = out.any_name
            print(f"\n   Output [{i}] '{name}':")
            print(f"     Shape: {data.shape}")
            print(f"     Dtype: {data.dtype}")
            print(f"     Range: [{data.min():.4f}, {data.max():.4f}]")

            # Analyze based on shape
            flat = data.flatten()
            print(f"     First 10 values: {flat[:10]}")

            # If looks like boxes (values in reasonable range)
            if data.max() <= 1.0 and data.min() >= 0:
                print(f"     Looks like NORMALIZED coordinates")
            elif data.max() > 100:
                print(f"     Looks like PIXEL coordinates (max={data.max():.1f})")

            # If looks like labels (integers or small range)
            if 'label' in name.lower() or 'class' in name.lower():
                unique = np.unique(flat[:1000].astype(int))
                print(f"     Unique label values: {unique[:20]}")
                print(f"     Label range: {int(data.min())} to {int(data.max())}")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check dataset
    print("\n5. Checking dataset...")
    data_paths = [
        Path("data/openimages"),
        Path.home() / "datasets" / "openimages",
    ]

    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break

    if not data_path:
        print("   Dataset not found!")
        print("   Expected paths:", [str(p) for p in data_paths])
    else:
        print(f"   Found: {data_path}")

        # Check for images
        images_dirs = [data_path / "images", data_path / "validation" / "data"]
        for img_dir in images_dirs:
            if img_dir.exists():
                num_images = len(list(img_dir.glob("*.jpg")))
                print(f"   Images in {img_dir}: {num_images}")

        # Check for annotations
        ann_files = list(data_path.rglob("*.json"))
        print(f"   JSON files found: {[str(f.name) for f in ann_files]}")

        for ann_file in ann_files:
            if "mlperf" in ann_file.name.lower():
                print(f"\n   Analyzing {ann_file.name}:")
                with open(ann_file) as f:
                    coco = json.load(f)
                print(f"     Categories: {len(coco.get('categories', []))}")
                print(f"     Images: {len(coco.get('images', []))}")
                print(f"     Annotations: {len(coco.get('annotations', []))}")

                cats = coco.get('categories', [])
                if cats:
                    cat_ids = [c['id'] for c in cats]
                    print(f"     Category ID range: {min(cat_ids)} to {max(cat_ids)}")
                    print(f"     First 5 categories: {cats[:5]}")

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
