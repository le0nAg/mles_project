# CMakeLists.txt for the processing library
cmake_minimum_required(VERSION 3.10)
project(processing C)

# Add the library target
add_library(testing STATIC testing.c)

# Specify the include directory for testing.h
target_include_directories(testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
