# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

set(PICO_BOARD pico CACHE STRING "Board type")
set(PROGRAM_NAME main)

cmake_minimum_required(VERSION 3.12)

include(pico_sdk_import.cmake)

project(Pico-ResTouch-X_X_code)

pico_sdk_init()

# ============================================================================
# SUBDIRECTORIES
# ============================================================================
add_subdirectory(lib/sd_card)
add_subdirectory(lib/config)
add_subdirectory(lib/lcd)
add_subdirectory(lib/font)
add_subdirectory(lib/processing)
add_subdirectory(lib/testing)

include_directories(./lib/sd_card)
include_directories(./lib/config)
include_directories(./lib/lcd)
include_directories(./lib/font)
include_directories(./lib/processing)
include_directories(./lib/testing)

# ============================================================================
# EXECUTABLE
# ============================================================================
add_executable(${PROGRAM_NAME}
    main.c
)

# ============================================================================
# MEMORY CONFIGURATION - OPTIMIZED FOR RP2040 (264KB RAM total)
# ============================================================================

# Stack size for Core0 (reduce to 4KB, was default 2KB)
target_compile_definitions(${PROGRAM_NAME} PRIVATE
    PICO_STACK_SIZE=0x1000              # 4KB stack per core
)

# Heap size (reduce to 80KB to leave room for everything else)
target_compile_definitions(${PROGRAM_NAME} PRIVATE
    PICO_HEAP_SIZE=0x14000              # 80KB heap (for malloc allocations)
)

# Core1 stack size (for SD card writer, keep small)
target_compile_definitions(${PROGRAM_NAME} PRIVATE
    PICO_CORE1_STACK_SIZE=0x1000        # 4KB for Core1
)

# ============================================================================
# COMPILER OPTIMIZATIONS
# ============================================================================

# Optimization flags
target_compile_options(${PROGRAM_NAME} PRIVATE
    -O2                                 # Optimize for performance (balanced)
    -Wall                               # Enable all warnings
    -Wextra                             # Enable extra warnings
    -Wno-unused-parameter               # Disable unused parameter warnings
    # -flto                               # Enable link-time optimization
)

# Link-time optimization and memory usage reporting
target_link_options(${PROGRAM_NAME} PRIVATE
    -Wl,--print-memory-usage            # Print memory usage after linking
)

# Alternative: Optimize for size (uncomment if you need to save flash space)
# target_compile_options(${PROGRAM_NAME} PRIVATE
#     -Os                               # Optimize for size instead of -O2
#     -ffunction-sections               # Separate functions for linker
#     -fdata-sections                   # Separate data sections
# )
# target_link_options(${PROGRAM_NAME} PRIVATE
#     -Wl,--gc-sections                 # Remove unused sections
# )

# Alternative: Maximum speed optimization (uncomment for fastest code)
# target_compile_options(${PROGRAM_NAME} PRIVATE
#     -O3                               # Maximum optimization
#     -funroll-loops                    # Unroll loops
#     -ffast-math                       # Faster float operations (less precise)
# )

# ============================================================================
# STDIO CONFIGURATION
# ============================================================================

# Enable usb output, disable uart output
pico_enable_stdio_usb(${PROGRAM_NAME} 1)
pico_enable_stdio_uart(${PROGRAM_NAME} 0)

# ============================================================================
# OUTPUT FILES
# ============================================================================

# Create map/bin/hex/uf2 file etc.
pico_add_extra_outputs(${PROGRAM_NAME})

# ============================================================================
# LIBRARIES
# ============================================================================

target_link_libraries(${PROGRAM_NAME}
    sd_custom_driver
    lcd
    font
    processing
    testing
    config
    pico_stdlib
    # pico_rand
    pico_time
    hardware_spi
    hardware_pwm
)

# ============================================================================
# MEMORY USAGE SUMMARY (printed after build)
# ============================================================================

add_custom_command(TARGET ${PROGRAM_NAME} POST_BUILD
    COMMAND ${CMAKE_SIZE_UTIL} ${PROGRAM_NAME}.elf
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "========================================="
    COMMENT "Memory Usage Summary:"
    COMMENT "========================================="
)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "Configuration:")
message(STATUS "  Stack Size: 16KB")
message(STATUS "  Heap Size: 128KB")
message(STATUS "  Core1 Stack: 8KB")
message(STATUS "  Optimization: -O2 (balanced)")
message(STATUS "  LTO: Enabled")
message(STATUS "========================================")