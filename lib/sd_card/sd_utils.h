#ifndef __SD_UTILS_H_
#define __SD_UTILS_H_

#include <stdint.h>
#include <stdbool.h>

void sd_writer_init(void);
void sd_writer_shutdown(void);
bool sd_write_async(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename);
bool sd_write_async_packed(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename);
uint32_t sd_writer_pending_count(void);
bool sd_writer_is_busy(void);
void generate_uuid(char out[37]);
bool sd_write_async_json(const char *json_str, const char *filename);

#endif