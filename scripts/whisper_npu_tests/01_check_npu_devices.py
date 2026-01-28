#!/usr/bin/env python3
"""
Step 1: Check available NPU devices and their properties.

This script discovers all NPU dies and prints their capabilities.
Run this first to understand your hardware configuration.

Usage:
    python 01_check_npu_devices.py
"""

import sys


def main():
    print("=" * 60)
    print("Step 1: NPU Device Discovery")
    print("=" * 60)

    # Check OpenVINO installation
    try:
        import openvino as ov
        print(f"\n✓ OpenVINO version: {ov.__version__}")
    except ImportError:
        print("\n✗ OpenVINO not installed!")
        print("  Install with: pip install openvino")
        sys.exit(1)

    core = ov.Core()

    # Get all available devices
    print("\n" + "-" * 40)
    print("Available devices:")
    print("-" * 40)

    devices = core.available_devices
    for device in devices:
        print(f"  - {device}")

    # Find NPU devices
    npu_devices = [d for d in devices if d.upper().startswith("NPU")]

    if not npu_devices:
        print("\n✗ No NPU devices found!")
        print("  Available devices:", devices)
        print("\n  If you have NPU hardware, check:")
        print("  1. NPU driver is installed")
        print("  2. OpenVINO NPU plugin is available")
        sys.exit(1)

    print(f"\n✓ Found {len(npu_devices)} NPU device(s): {npu_devices}")

    # Get properties for each NPU device
    print("\n" + "-" * 40)
    print("NPU Device Properties:")
    print("-" * 40)

    # Key properties to check
    important_props = [
        "FULL_DEVICE_NAME",
        "DEVICE_ARCHITECTURE",
        "OPTIMIZATION_CAPABILITIES",
        "RANGE_FOR_ASYNC_INFER_REQUESTS",
        "RANGE_FOR_STREAMS",
        "SUPPORTED_PROPERTIES",
    ]

    for npu_device in npu_devices:
        print(f"\n[{npu_device}]")

        try:
            # Get full device name
            try:
                full_name = core.get_property(npu_device, "FULL_DEVICE_NAME")
                print(f"  Full name: {full_name}")
            except Exception:
                pass

            # Get architecture
            try:
                arch = core.get_property(npu_device, "DEVICE_ARCHITECTURE")
                print(f"  Architecture: {arch}")
            except Exception:
                pass

            # Get optimization capabilities
            try:
                caps = core.get_property(npu_device, "OPTIMIZATION_CAPABILITIES")
                print(f"  Optimization capabilities: {caps}")
            except Exception:
                pass

            # Get supported properties list
            try:
                supported = core.get_property(npu_device, "SUPPORTED_PROPERTIES")
                print(f"  Supported properties count: {len(supported)}")
            except Exception:
                pass

        except Exception as e:
            print(f"  Error getting properties: {e}")

    # Check for multi-die configuration
    print("\n" + "-" * 40)
    print("Multi-Die Analysis:")
    print("-" * 40)

    # Count dies (NPU.0, NPU.1, etc.)
    specific_dies = [d for d in npu_devices if "." in d]
    generic_npu = [d for d in npu_devices if d.upper() == "NPU"]

    if specific_dies:
        print(f"\n✓ Found {len(specific_dies)} specific NPU dies: {specific_dies}")
        print("  Multi-die inference is supported!")

    if generic_npu:
        print(f"\n✓ Generic 'NPU' device available")
        print("  This will use all available dies automatically")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Total NPU devices: {len(npu_devices)}")
    print(f"  Specific dies: {len(specific_dies)}")
    print(f"  Generic NPU: {'Yes' if generic_npu else 'No'}")

    if len(specific_dies) >= 2:
        print("\n✓ Multi-die configuration detected!")
        print("  You can distribute workloads across dies.")

    print("\n" + "=" * 60)
    print("Next step: Run 02_check_optimum_npu.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
