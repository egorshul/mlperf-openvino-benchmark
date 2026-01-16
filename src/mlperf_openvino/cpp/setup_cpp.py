"""
Build script for C++ SUT extension.

Usage:
    python setup_cpp.py build_ext --inplace

Or to install:
    pip install .
"""

import os
import sys
import subprocess
from pathlib import Path

def build_extension():
    """Build the C++ extension using CMake."""
    cpp_dir = Path(__file__).parent
    build_dir = cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Configure
    cmake_args = [
        "cmake",
        str(cpp_dir),
        f"-DCMAKE_INSTALL_PREFIX={cpp_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    # Add Python executable
    cmake_args.append(f"-DPython3_EXECUTABLE={sys.executable}")

    print(f"Configuring: {' '.join(cmake_args)}")
    subprocess.check_call(cmake_args, cwd=build_dir)

    # Build
    build_args = ["cmake", "--build", ".", "--config", "Release", "-j"]
    print(f"Building: {' '.join(build_args)}")
    subprocess.check_call(build_args, cwd=build_dir)

    # Install (copy to package directory)
    install_args = ["cmake", "--install", "."]
    print(f"Installing: {' '.join(install_args)}")
    subprocess.check_call(install_args, cwd=build_dir)

    print("Build complete!")


if __name__ == "__main__":
    build_extension()
