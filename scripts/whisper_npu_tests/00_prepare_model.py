#!/usr/bin/env python3
"""
Step 0: Prepare Whisper model for NPU inference.

This script exports Whisper model to OpenVINO format using optimum-cli.
Run this before the other tests if you don't have the model yet.

Usage:
    python 00_prepare_model.py [--output-dir ./models/whisper-large-v3-openvino]

Requirements:
    pip install optimum[openvino] transformers
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    print("[1/3] Checking dependencies...")

    missing = []

    try:
        import optimum
        try:
            from optimum.version import __version__ as optimum_version
        except ImportError:
            optimum_version = "unknown"
        print(f"  ✓ optimum: {optimum_version}")
    except ImportError:
        missing.append("optimum[openvino]")
        print("  ✗ optimum not installed")

    try:
        import transformers
        print(f"  ✓ transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
        print("  ✗ transformers not installed")

    try:
        import openvino
        print(f"  ✓ openvino: {openvino.__version__}")
    except ImportError:
        missing.append("openvino")
        print("  ✗ openvino not installed")

    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def export_model(model_id: str, output_dir: Path):
    """Export Whisper model to OpenVINO format."""
    print(f"\n[2/3] Exporting model: {model_id}")
    print(f"       Output: {output_dir}")

    # Try optimum-cli first, fallback to python -m
    use_cli = False
    try:
        result = subprocess.run(
            ["optimum-cli", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            use_cli = True
            print("  ✓ optimum-cli found")
    except FileNotFoundError:
        pass

    if not use_cli:
        print("  optimum-cli not in PATH, using python -m optimum.exporters.openvino")

    # Export command
    # --task automatic-speech-recognition-with-past creates:
    #   - encoder_model.xml
    #   - decoder_model.xml (first step, no cache)
    #   - decoder_with_past_model.xml (subsequent steps, with KV cache)
    if use_cli:
        cmd = [
            "optimum-cli", "export", "openvino",
            "--model", model_id,
            "--task", "automatic-speech-recognition-with-past",
            str(output_dir)
        ]
    else:
        cmd = [
            sys.executable, "-m", "optimum.exporters.openvino",
            "--model", model_id,
            "--task", "automatic-speech-recognition-with-past",
            str(output_dir)
        ]

    print(f"\n  Running: {' '.join(cmd)}")
    print("  This may take several minutes...\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            print(f"  {line}", end="")

        process.wait()

        if process.returncode != 0:
            print(f"\n  ✗ Export failed with code {process.returncode}")
            return False

        # Verify files were actually created
        encoder_file = output_dir / "encoder_model.xml"
        if not encoder_file.exists():
            # Try alternative: openvino_encoder_model.xml (newer optimum versions)
            encoder_file = output_dir / "openvino_encoder_model.xml"

        if not encoder_file.exists():
            print(f"\n  ✗ Export command succeeded but no model files found!")
            print(f"  Check if model was downloaded and exported correctly.")
            print(f"  You may need to run manually:")
            print(f"    {' '.join(cmd)}")
            return False

        print(f"\n  ✓ Export completed!")
        return True

    except Exception as e:
        print(f"\n  ✗ Export failed: {e}")
        return False


def verify_export(output_dir: Path):
    """Verify the exported model files."""
    print(f"\n[3/3] Verifying export...")

    expected_files = [
        "encoder_model.xml",
        "encoder_model.bin",
        "decoder_with_past_model.xml",
        "decoder_with_past_model.bin",
    ]

    optional_files = [
        "decoder_model.xml",
        "decoder_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ]

    all_ok = True

    for f in expected_files:
        path = output_dir / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f} NOT FOUND")
            all_ok = False

    print("\n  Optional files:")
    for f in optional_files:
        path = output_dir / f
        if path.exists():
            print(f"    ✓ {f}")
        else:
            print(f"    - {f} (missing)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Whisper model for NPU inference"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/whisper-large-v3-openvino",
        help="Output directory for OpenVINO model"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export even if model exists"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 0: Prepare Whisper Model for NPU")
    print("=" * 60)

    output_dir = Path(args.output_dir)

    # Check if model already exists
    if not args.force and (output_dir / "encoder_model.xml").exists():
        print(f"\n  Model already exists at {output_dir}")
        print("  Use --force to re-export")

        if verify_export(output_dir):
            print("\n" + "=" * 60)
            print("Model is ready! Run the test scripts:")
            print("=" * 60)
            print(f"\n  python 01_check_npu_devices.py")
            print(f"  python 02_check_optimum_npu.py --model-path {output_dir}")
            return 0
        else:
            print("\n  Existing export is incomplete, consider --force")
            return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export model
    if not export_model(args.model_id, output_dir):
        return 1

    # Verify export
    if not verify_export(output_dir):
        return 1

    print("\n" + "=" * 60)
    print("Model exported successfully!")
    print("=" * 60)
    print(f"\n  Model location: {output_dir}")
    print("\n  Next steps:")
    print(f"  1. python 01_check_npu_devices.py")
    print(f"  2. python 02_check_optimum_npu.py --model-path {output_dir}")
    print(f"  3. python 03_test_manual_encoder_decoder.py --model-path {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
