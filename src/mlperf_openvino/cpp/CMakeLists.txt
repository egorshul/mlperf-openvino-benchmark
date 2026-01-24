cmake_minimum_required(VERSION 3.14)
project(mlperf_ov_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include for downloading dependencies
include(FetchContent)

# =============================================================================
# Download MLPerf LoadGen source only (not entire inference repo)
# =============================================================================
set(LOADGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/loadgen")

if(NOT EXISTS "${LOADGEN_DIR}/CMakeLists.txt")
    message(STATUS "Downloading MLPerf LoadGen source...")

    # Download specific files from GitHub (much faster than cloning)
    set(LOADGEN_URL "https://raw.githubusercontent.com/mlcommons/inference/master/loadgen")

    file(MAKE_DIRECTORY "${LOADGEN_DIR}")

    # All loadgen source and header files (from mlcommons/inference/loadgen)
    set(LOADGEN_FILES
        # Headers
        "bindings/c_api.h"
        "early_stopping.h"
        "issue_query_controller.h"
        "loadgen.h"
        "logging.h"
        "query_dispatch_library.h"
        "query_sample.h"
        "query_sample_library.h"
        "results.h"
        "system_under_test.h"
        "test_settings.h"
        "test_settings_internal.h"
        "utils.h"
        "version.h"
        # Sources
        "bindings/c_api.cc"
        "early_stopping.cc"
        "issue_query_controller.cc"
        "loadgen.cc"
        "logging.cc"
        "results.cc"
        "test_settings_internal.cc"
        "utils.cc"
        "version.cc"
        # Config file (used to generate mlperf_conf.h)
        "mlperf.conf"
    )

    file(MAKE_DIRECTORY "${LOADGEN_DIR}/bindings")

    # Count files for progress
    list(LENGTH LOADGEN_FILES TOTAL_FILES)
    set(CURRENT_FILE 0)

    foreach(FILE ${LOADGEN_FILES})
        math(EXPR CURRENT_FILE "${CURRENT_FILE} + 1")
        if(NOT EXISTS "${LOADGEN_DIR}/${FILE}")
            message(STATUS "  [${CURRENT_FILE}/${TOTAL_FILES}] Downloading ${FILE}...")
            file(DOWNLOAD
                "${LOADGEN_URL}/${FILE}"
                "${LOADGEN_DIR}/${FILE}"
                STATUS DOWNLOAD_STATUS
                TIMEOUT 30
                SHOW_PROGRESS
            )
            list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
            if(NOT STATUS_CODE EQUAL 0)
                list(GET DOWNLOAD_STATUS 1 ERROR_MSG)
                message(FATAL_ERROR "Failed to download ${FILE}: ${ERROR_MSG}")
            endif()
        else()
            message(STATUS "  [${CURRENT_FILE}/${TOTAL_FILES}] ${FILE} (cached)")
        endif()
    endforeach()

    # Generate mlperf_conf.h from mlperf.conf
    message(STATUS "Generating mlperf_conf.h...")
    file(READ "${LOADGEN_DIR}/mlperf.conf" MLPERF_CONF_CONTENT)
    # Escape special characters for C++ string
    string(REPLACE "\\" "\\\\" MLPERF_CONF_CONTENT "${MLPERF_CONF_CONTENT}")
    string(REPLACE "\"" "\\\"" MLPERF_CONF_CONTENT "${MLPERF_CONF_CONTENT}")
    string(REPLACE "\n" "\\n\"\n\"" MLPERF_CONF_CONTENT "${MLPERF_CONF_CONTENT}")
    file(WRITE "${LOADGEN_DIR}/mlperf_conf.h"
"// Auto-generated from mlperf.conf
#ifndef MLPERF_CONF_H
#define MLPERF_CONF_H
static const char mlperf_conf[] = \"${MLPERF_CONF_CONTENT}\";
#endif // MLPERF_CONF_H
")

    # Generate version_generated.cc with all required functions
    message(STATUS "Generating version_generated.cc...")
    string(TIMESTAMP BUILD_DATE_LOCAL "%Y-%m-%dT%H:%M:%S")
    string(TIMESTAMP BUILD_DATE_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)
    file(WRITE "${LOADGEN_DIR}/version_generated.cc"
"// Auto-generated version info (replaces version_generator.py output)
#include <string>

namespace mlperf {

const std::string& LoadgenVersion() {
    static const std::string version = \"v4.1\";
    return version;
}

const std::string& LoadgenGitRevision() {
    static const std::string revision = \"cmake-build\";
    return revision;
}

const std::string& LoadgenBuildDateLocal() {
    static const std::string date = \"${BUILD_DATE_LOCAL}\";
    return date;
}

const std::string& LoadgenBuildDateUtc() {
    static const std::string date = \"${BUILD_DATE_UTC}\";
    return date;
}

const std::string& LoadgenGitCommitDate() {
    static const std::string date = \"NA\";
    return date;
}

const std::string& LoadgenGitStatus() {
    static const std::string status = \"NA\";
    return status;
}

const std::string& LoadgenGitLog() {
    static const std::string log = \"Built from source via CMake\";
    return log;
}

const std::string& LoadgenSha1OfFiles() {
    static const std::string sha1 = \"NA NA\";
    return sha1;
}

}  // namespace mlperf
")

    # Create CMakeLists.txt for loadgen
    file(WRITE "${LOADGEN_DIR}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.14)
project(mlperf_loadgen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mlperf_loadgen STATIC
    bindings/c_api.cc
    early_stopping.cc
    issue_query_controller.cc
    loadgen.cc
    logging.cc
    results.cc
    test_settings_internal.cc
    utils.cc
    version.cc
    version_generated.cc
)

target_include_directories(mlperf_loadgen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
]=])

    message(STATUS "MLPerf LoadGen downloaded to ${LOADGEN_DIR}")
endif()

# Build loadgen
add_subdirectory("${LOADGEN_DIR}" "${CMAKE_BINARY_DIR}/loadgen_build")

message(STATUS "MLPerf LoadGen configured")

# =============================================================================
# Find OpenVINO from Python package
# =============================================================================
execute_process(
    COMMAND python3 -c "import openvino; import os; print(os.path.dirname(openvino.__file__))"
    OUTPUT_VARIABLE OPENVINO_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(OpenVINO_DIR "${OPENVINO_PYTHON_DIR}/cmake")
message(STATUS "OpenVINO_DIR: ${OpenVINO_DIR}")

# =============================================================================
# Find pybind11 from Python package
# =============================================================================
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

# =============================================================================
# Create the Python extension module
# =============================================================================
pybind11_add_module(_cpp_sut
    resnet_sut_cpp.cpp
    bert_sut_cpp.cpp
    retinanet_sut_cpp.cpp
    resnet_multi_die_sut_cpp.cpp
    bindings.cpp
)

# Link libraries
target_link_libraries(_cpp_sut PRIVATE
    openvino::runtime
    mlperf_loadgen
)

# Include directories
target_include_directories(_cpp_sut PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LOADGEN_DIR}
)

# Enable direct LoadGen C++ integration
target_compile_definitions(_cpp_sut PRIVATE USE_LOADGEN_CPP=1)

# =============================================================================
# Optimization flags
# =============================================================================
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

# =============================================================================
# Install
# =============================================================================
install(TARGETS _cpp_sut
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)

