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

# Build option: OFF = main application, ON = digit collector
option(BUILD_COLLECTOR "Build the digit collector application instead of main app" OFF)

if(BUILD_COLLECTOR)
    set(PROGRAM_NAME digit_collector)
else()
    set(PROGRAM_NAME pin_auth)
endif()

cmake_minimum_required(VERSION 3.12)

include(pico_sdk_import.cmake)

project(Pico-ResTouch-X_X_code)

pico_sdk_init()

add_subdirectory(lib/sd_card)
add_subdirectory(lib/config)
add_subdirectory(lib/lcd)
add_subdirectory(lib/font)
add_subdirectory(lib/processing)
add_subdirectory(lib/testing)
add_subdirectory(lib/ml)

include_directories(./lib/sd_card)
include_directories(./lib/config)
include_directories(./lib/lcd)
include_directories(./lib/font)
include_directories(./lib/processing)
include_directories(./lib/testing)
# FIX: auth_ui is under lib/, not in root
include_directories(./lib/auth_ui)
include_directories(./digit_collector)

if(BUILD_COLLECTOR)
    add_executable(${PROGRAM_NAME}
        main_collector.c
        digit_collector/digit_collector.c
    )
else()
    add_executable(${PROGRAM_NAME}
        main.c
        # FIX: auth_ui is under lib/
        lib/auth_ui/pin_auth_ui.c
    )
endif()

target_compile_definitions(${PROGRAM_NAME} PRIVATE
    PICO_STACK_SIZE=0x1000
    PICO_HEAP_SIZE=0x14000
    PICO_CORE1_STACK_SIZE=0x1000
)

target_compile_options(${PROGRAM_NAME} PRIVATE
    -O2
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_link_options(${PROGRAM_NAME} PRIVATE
    -Wl,--print-memory-usage
)

pico_enable_stdio_usb(${PROGRAM_NAME} 1)
pico_enable_stdio_uart(${PROGRAM_NAME} 0)

pico_add_extra_outputs(${PROGRAM_NAME})

target_link_libraries(${PROGRAM_NAME}
    sd_custom_driver
    lcd
    font
    processing
    testing
    ml
    config
    pico_stdlib
    pico_time
    hardware_spi
    hardware_pwm
)