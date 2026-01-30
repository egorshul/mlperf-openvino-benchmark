cmake_minimum_required(VERSION 3.16)
project(convert_decoder_static CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(convert_decoder_static main.cpp)
target_link_libraries(convert_decoder_static PRIVATE openvino::runtime)
