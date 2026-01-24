# C++ SUT for MLPerf Inference

High-performance C++ System Under Test (SUT) implementation for MLPerf Inference benchmarks.

## Key Features

- **Direct LoadGen C++ integration** - Like NVIDIA LWIS, calls `mlperf::QuerySamplesComplete` directly from C++ without Python/GIL overhead
- **Multi-die accelerator support** - Automatically discovers and utilizes all available NPU/VPU dies
- **Zero-copy data registration** - Pre-register QSL data pointers in C++ for maximum Server mode throughput
- **Async inference pipelining** - Multiple inference requests per die for optimal hardware utilization

## Architecture

```
Python (LoadGen)
    │
    ├─[issue_queries]─► Python Wrapper ─► C++ SUT
    │                                        │
    │                                   [inference]
    │                                        │
    └───────────────────────────────────◄────┘
                                  [mlperf::QuerySamplesComplete]
                                   (direct C++ call, NO GIL!)
```

## Building

The build automatically downloads MLPerf LoadGen from GitHub:

```bash
cd /home/user/mlperf-openvino-benchmark/src/mlperf_openvino/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
cp _cpp_sut*.so ..
```

### Requirements

- CMake 3.14+
- C++17 compiler (GCC 7+ or Clang 5+)
- OpenVINO (Python package)
- pybind11 (Python package)
- Git (for automatic LoadGen download)

## Performance

Server mode performance comparison (ResNet50, 60000 QPS target):

| Mode | GIL Acquires/sec | Expected QPS |
|------|-----------------|--------------|
| Python callback | 120,000 | ~17,000 |
| **Direct LoadGen C++** | **0** | **~55,000** |

## Files

- `resnet_multi_die_sut_cpp.hpp/cpp` - Multi-die SUT for ResNet50
- `bindings.cpp` - pybind11 Python bindings
- `CMakeLists.txt` - Build configuration

## Usage

```python
from mlperf_openvino.cpp import ResNetMultiDieCppSUT

# Create SUT
sut = ResNetMultiDieCppSUT(
    model_path="/path/to/resnet50.onnx",
    device_prefix="NPU",
    batch_size=1,
    compile_properties={},
    use_nhwc_input=True
)

# Load model
sut.load()

# Enable direct LoadGen for Server mode
sut.enable_direct_loadgen(True)

# Register QSL data for fast dispatch
for idx, data in loaded_samples.items():
    sut.register_sample_data(idx, data)

# Fast dispatch (only query_ids and sample_indices, no data copying)
sut.issue_queries_server_fast(query_ids, sample_indices)
```
