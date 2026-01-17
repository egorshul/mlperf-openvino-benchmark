#!/bin/bash
# Build C++ SUT extension for maximum throughput

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="${SCRIPT_DIR}/src/mlperf_openvino/cpp"
BUILD_DIR="${CPP_DIR}/build"

echo "Building C++ SUT extension..."
echo "CPP_DIR: ${CPP_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "Configuring..."
cmake "${CPP_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CPP_DIR}" \
    -DPython3_EXECUTABLE=$(which python3)

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc)

# Install (copy to package directory)
echo "Installing..."
cmake --install .

echo ""
echo "Build complete!"
echo "C++ extension installed to: ${CPP_DIR}"
echo ""
echo "Test with:"
echo "  python -c 'from mlperf_openvino.cpp import ResNetCppSUT, CPP_AVAILABLE; print(f\"C++ SUT available: {CPP_AVAILABLE}\")'"
