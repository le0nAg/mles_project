cmake_minimum_required(VERSION 3.12)

# Fetch pico-tflmicro
include(FetchContent)
FetchContent_Declare(
    pico-tflmicro
    GIT_REPOSITORY https://github.com/raspberrypi/pico-tflmicro.git
    GIT_TAG main
)
FetchContent_MakeAvailable(pico-tflmicro)

# ML inference library
add_library(ml INTERFACE)

target_sources(ml INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/digit_inference.cc
)

target_include_directories(ml INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ml INTERFACE
    pico-tflmicro
)
