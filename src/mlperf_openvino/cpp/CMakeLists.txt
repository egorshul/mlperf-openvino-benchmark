cmake_minimum_required(VERSION 3.14)
project(mlperf_ov_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to enable direct LoadGen C++ integration (like NVIDIA)
option(USE_LOADGEN_CPP "Enable direct LoadGen C++ calls (requires libmlperf_loadgen)" OFF)

# Find OpenVINO from Python package
execute_process(
    COMMAND python3 -c "import openvino; import os; print(os.path.dirname(openvino.__file__))"
    OUTPUT_VARIABLE OPENVINO_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(OpenVINO_DIR "${OPENVINO_PYTHON_DIR}/cmake")
message(STATUS "OpenVINO_DIR: ${OpenVINO_DIR}")

# Find pybind11 from Python package
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "pybind11_DIR: ${pybind11_DIR}")

# Find required packages
find_package(OpenVINO REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Create the Python extension module
pybind11_add_module(_cpp_sut
    resnet_sut_cpp.cpp
    bert_sut_cpp.cpp
    retinanet_sut_cpp.cpp
    resnet_multi_die_sut_cpp.cpp
    retinanet_multi_die_sut_cpp.cpp
    bindings.cpp
)

# Link OpenVINO
target_link_libraries(_cpp_sut PRIVATE openvino::runtime)

# Include directories
target_include_directories(_cpp_sut PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: Direct LoadGen C++ integration for maximum performance
if(USE_LOADGEN_CPP)
    message(STATUS "Enabling direct LoadGen C++ integration")

    # Find LoadGen library
    find_library(MLPERF_LOADGEN_LIB
        NAMES mlperf_loadgen
        PATHS /usr/local/lib /usr/lib
    )

    # Find LoadGen headers
    find_path(MLPERF_LOADGEN_INCLUDE
        NAMES query_sample.h
        PATHS /usr/local/include /usr/include
    )

    if(MLPERF_LOADGEN_LIB AND MLPERF_LOADGEN_INCLUDE)
        message(STATUS "LoadGen library: ${MLPERF_LOADGEN_LIB}")
        message(STATUS "LoadGen include: ${MLPERF_LOADGEN_INCLUDE}")

        target_link_libraries(_cpp_sut PRIVATE ${MLPERF_LOADGEN_LIB})
        target_include_directories(_cpp_sut PRIVATE ${MLPERF_LOADGEN_INCLUDE})
        target_compile_definitions(_cpp_sut PRIVATE USE_LOADGEN_CPP=1)
    else()
        message(WARNING "LoadGen C++ not found! Install from: https://github.com/mlcommons/inference/tree/master/loadgen")
        message(WARNING "  git clone https://github.com/mlcommons/inference.git")
        message(WARNING "  cd inference/loadgen && mkdir build && cd build")
        message(WARNING "  cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make && sudo make install")
    endif()
endif()

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_cpp_sut PRIVATE
        -O3
        -march=native
        -ffast-math
    )
elseif(MSVC)
    target_compile_options(_cpp_sut PRIVATE
        /O2
        /fp:fast
    )
endif()

# Install to the package directory
install(TARGETS _cpp_sut
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
