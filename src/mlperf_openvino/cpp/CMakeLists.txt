cmake_minimum_required(VERSION 3.14)
project(mlperf_ov_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    sut_cpp.cpp
    offline_sut.cpp
    bert_sut_cpp.cpp
    retinanet_sut_cpp.cpp
    whisper_sut_cpp.cpp
    bindings.cpp
)

# Link OpenVINO
target_link_libraries(_cpp_sut PRIVATE openvino::runtime)

# Include directories
target_include_directories(_cpp_sut PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
