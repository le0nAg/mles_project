aux_source_directory(. DIR_LCD_SRCS)

include_directories(../config)
include_directories(../font)
include_directories(../sd_card)

add_library(lcd ${DIR_LCD_SRCS})
target_link_libraries(lcd PUBLIC config font sd_custom_driver pico_stdlib)