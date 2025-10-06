# CMakeLists.txt for the processing library
cmake_minimum_required(VERSION 3.10)
project(processing C)

# Add the library target
add_library(processing STATIC processing.c)

# Specify the include directory for processing.h
target_include_directories(processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
