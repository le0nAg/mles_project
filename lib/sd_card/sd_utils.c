#include "sd_utils.h"
#include "ff.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "pico/mutex.h"
#include "hw_config.h"

#define PATH_MAX_LEN 256


/*
The program is going to use always the core1 to write on the SD for two reasons:
1) library limitations
2) code optimization: in this way core 0 can care about data processing

TODO: timestamp the data
*/

// ============================================================================
// CONCURRENCY MANAGEMENT
// ============================================================================

typedef enum {
    SD_CMD_WRITE_TEXT,
    SD_CMD_WRITE_BINARY,
    SD_CMD_SHUTDOWN
} SD_Command_Type;

typedef struct {
    SD_Command_Type cmd_type;
    uint8_t *bitmap;
    uint16_t width;
    uint16_t height;
    char filename[64];
} SD_Write_Command;

static queue_t sd_command_queue;
static mutex_t bitmap_mutex;

#define MAX_BITMAP_SIZE (320 * 240)
static uint8_t shared_bitmap_buffer[MAX_BITMAP_SIZE];

static volatile bool core1_running = false;

// ============================================================================
// FILE SYSTEM STATE (mounted once, kept alive)
// ============================================================================

static FATFS fs;
static sd_card_t *g_sd = NULL;
static const char *g_drive = NULL;
static bool fs_mounted = false;

// ============================================================================
// PRIVATE FUNCTIONS - SD CARD OPERATIONS
// ============================================================================

/**
 * Initialize SD card driver and mount filesystem (called once)
 * @return true on success, false on failure
 */
static bool sd_init_and_mount(void)
{
    if (fs_mounted) {
        printf("SD: Already mounted\r\n");
        return true;
    }

    // Initialize the SD card driver
    if (!sd_init_driver()) {
        printf("ERROR: sd_init_driver() failed\r\n");
        return false;
    }

    // Get SD card configuration
    g_sd = sd_get_by_num(0);
    if (!g_sd) {
        printf("ERROR: No SD config found (sd_get_by_num(0) == NULL)\r\n");
        return false;
    }

    // Get drive prefix (usually "0:")
    g_drive = sd_get_drive_prefix(g_sd);
    if (!g_drive) {
        printf("ERROR: sd_get_drive_prefix() returned NULL\r\n");
        return false;
    }

    // Mount the filesystem
    FRESULT fr = f_mount(&fs, g_drive, 1);
    printf("SD: f_mount -> %s (%d)\r\n", FRESULT_str(fr), fr);

    // If no filesystem exists, create one
    // if (fr == FR_NO_FILESYSTEM) {
    //     printf("SD: No filesystem found, creating FAT...\r\n");
    //     BYTE work[4096];
    //     MKFS_PARM opt = { FM_FAT | FM_SFD, 0, 0, 0, 0 };
    //     fr = f_mkfs(g_drive, &opt, work, sizeof(work));
    //     printf("SD: f_mkfs -> %s (%d)\r\n", FRESULT_str(fr), fr);
        
    //     if (fr == FR_OK) {
    //         fr = f_mount(&fs, g_drive, 1);
    //         printf("SD: f_mount (after mkfs) -> %s (%d)\r\n", FRESULT_str(fr), fr);
    //     }
    // }

    if (fr != FR_OK) {
        printf("ERROR: Mount failed: %s (%d)\r\n", FRESULT_str(fr), fr);
        return false;
    }

    fs_mounted = true;
    printf("SD: Mounted successfully on '%s'\r\n", g_drive);
    return true;
}

/**
 * Create or open a file for writing
 * @param filename Filename relative to drive (e.g., "drawing.txt")
 * @param out_file Pointer to FIL structure to initialize
 * @return FRESULT status
 */
static FRESULT sd_create_file(const char *filename, FIL *out_file)
{
    if (!fs_mounted) {
        printf("ERROR: Filesystem not mounted\r\n");
        return FR_NOT_READY;
    }

    // Build full path: "0:/filename"
    char full_path[80];
    snprintf(full_path, sizeof(full_path), "%s/%s", g_drive, filename);

    FRESULT fr = f_open(out_file, full_path, FA_WRITE | FA_CREATE_ALWAYS);
    if (fr != FR_OK) {
        printf("ERROR: Failed to create file '%s': %s (%d)\r\n", 
               full_path, FRESULT_str(fr), fr);
    } else {
        printf("SD: Created file '%s'\r\n", full_path);
    }

    return fr;
}

/**
 * Write data to an open file
 * @param file Pointer to open FIL structure
 * @param data Data to write
 * @param len Length of data
 * @param bytes_written Pointer to receive number of bytes written
 * @return FRESULT status
 */
static FRESULT sd_write_data(FIL *file, const void *data, UINT len, UINT *bytes_written)
{
    *bytes_written = 0;
    
    FRESULT fr = f_write(file, data, len, bytes_written);
    if (fr != FR_OK) {
        printf("ERROR: f_write failed: %s (%d)\r\n", FRESULT_str(fr), fr);
        return fr;
    }

    if (*bytes_written != len) {
        printf("WARNING: Partial write: %u of %u bytes\r\n", *bytes_written, len);
    }

    // Sync to ensure data is written to card
    fr = f_sync(file);
    if (fr != FR_OK) {
        printf("ERROR: f_sync failed: %s (%d)\r\n", FRESULT_str(fr), fr);
    }

    return fr;
}

/**
 * Close an open file
 * @param file Pointer to open FIL structure
 * @return FRESULT status
 */
static FRESULT sd_close_file(FIL *file)
{
    FRESULT fr = f_close(file);
    if (fr != FR_OK) {
        printf("ERROR: f_close failed: %s (%d)\r\n", FRESULT_str(fr), fr);
    }
    return fr;
}

// ============================================================================
// WRITE IMPLEMENTATIONS
// ============================================================================

static void join_path(char *out, size_t out_sz, const char *drive, const char *rel) {
    // drive = "0:" or "0:/", ensure exactly one slash when joining
    if (rel && rel[0] == '/') rel++; // avoid double slashes
    if (drive && drive[strlen(drive) - 1] == '/')
        snprintf(out, out_sz, "%s%s", drive, rel ? rel : "");
    else
        snprintf(out, out_sz, "%s/%s", drive, rel ? rel : "");
}


/**
 * Write bitmap as ASCII '0' and '1' characters
 */
bool write_on_sd(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename)
{
    FIL fil;
    FRESULT fr;
    UINT bw;

    // Step 1: Create file
    char path[PATH_MAX_LEN];

    join_path(path, sizeof path, g_drive, "test3.txt");

    fr = sd_create_file(path, &fil);
    if (fr != FR_OK) {
        return false;
    }

    // Step 2: Allocate line buffer
    char *line = (char *)malloc(width + 2);
    if (!line) {
        printf("ERROR: Failed to allocate line buffer\r\n");
        sd_close_file(&fil);
        return false;
    }

    line[width] = '\n';
    line[width + 1] = '\0';

    // Step 3: Write data line by line
    bool success = true;
    for (uint16_t y = 0; y < height; y++) {
        // Convert binary to ASCII
        for (uint16_t x = 0; x < width; x++) {
            line[x] = bitmap[y * width + x] ? '1' : '0';
        }

        // Write line
        fr = sd_write_data(&fil, line, width + 1, &bw);
        if (fr != FR_OK || bw != width + 1) {
            printf("ERROR: Failed to write line %u\r\n", y);
            success = false;
            break;
        }
    }

    // Step 4: Cleanup
    free(line);
    sd_close_file(&fil);

    if (success) {
        printf("SUCCESS: Wrote %u x %u bitmap to '%s' (%u bytes)\r\n",
               width, height, filename, (width + 1) * height);
    }

    return success;
}

/**
 * Write bitmap in bit-packed binary format
 */
bool write_on_sd_bit_pack(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename)
{
    FIL fil;
    FRESULT fr;
    UINT bw;

    // Step 1: Create file
    fr = sd_create_file(filename, &fil);
    if (fr != FR_OK) {
        return false;
    }

    // Step 2: Write header (width + height)
    uint8_t header[4];
    header[0] = width & 0xFF;
    header[1] = (width >> 8) & 0xFF;
    header[2] = height & 0xFF;
    header[3] = (height >> 8) & 0xFF;

    fr = sd_write_data(&fil, header, 4, &bw);
    if (fr != FR_OK || bw != 4) {
        printf("ERROR: Failed to write header\r\n");
        sd_close_file(&fil);
        return false;
    }

    // Step 3: Allocate packed row buffer
    uint16_t bytes_per_row = (width + 7) / 8;
    uint8_t *packed_row = (uint8_t *)malloc(bytes_per_row);
    if (!packed_row) {
        printf("ERROR: Failed to allocate packed row buffer\r\n");
        sd_close_file(&fil);
        return false;
    }

    // Step 4: Write packed data row by row
    bool success = true;
    for (uint16_t y = 0; y < height; y++) {
        memset(packed_row, 0, bytes_per_row);

        // Pack 8 pixels per byte
        for (uint16_t x = 0; x < width; x++) {
            if (bitmap[y * width + x]) {
                uint16_t byte_idx = x / 8;
                uint8_t bit_idx = 7 - (x % 8);
                packed_row[byte_idx] |= (1 << bit_idx);
            }
        }

        // Write packed row
        fr = sd_write_data(&fil, packed_row, bytes_per_row, &bw);
        if (fr != FR_OK || bw != bytes_per_row) {
            printf("ERROR: Failed to write packed row %u\r\n", y);
            success = false;
            break;
        }
    }

    // Step 5: Cleanup
    free(packed_row);
    sd_close_file(&fil);

    if (success) {
        uint32_t total_bytes = 4 + bytes_per_row * height;
        float compression = (float)(width * height) / (float)total_bytes;
        printf("SUCCESS: Wrote %u x %u bitmap to '%s' (%u bytes, %.1fx compression)\r\n",
               width, height, filename, total_bytes, compression);
    }

    return success;
}

// ============================================================================
// CORE1 WORKER FUNCTION
// ============================================================================

static void core1_sd_writer(void)
{
    printf("Core1: SD writer started\r\n");
    
    // Initialize and mount SD card once on core1
    if (!sd_init_and_mount()) {
        printf("Core1: FATAL - Failed to initialize SD card\r\n");
        return;
    }
    
    core1_running = true;

    while (1) {
        SD_Write_Command cmd;

        // Wait for command from core0
        queue_remove_blocking(&sd_command_queue, &cmd);

        // Check for shutdown
        if (cmd.cmd_type == SD_CMD_SHUTDOWN) {
            printf("Core1: Shutdown requested\r\n");
            break;
        }

        // Copy bitmap data from shared buffer
        uint32_t bitmap_size = cmd.width * cmd.height;
        uint8_t *local_bitmap = (uint8_t *)malloc(bitmap_size);

        if (!local_bitmap) {
            printf("Core1: ERROR - Failed to allocate bitmap buffer\r\n");
            continue;
        }

        mutex_enter_blocking(&bitmap_mutex);
        memcpy(local_bitmap, shared_bitmap_buffer, bitmap_size);
        mutex_exit(&bitmap_mutex);

        // Process command
        bool success = false;

        switch (cmd.cmd_type) {
            case SD_CMD_WRITE_TEXT:
                printf("Core1: Writing text file '%s'...\r\n", cmd.filename);
                success = write_on_sd(local_bitmap, cmd.width, cmd.height, cmd.filename);
                break;

            case SD_CMD_WRITE_BINARY:
                printf("Core1: Writing binary file '%s'...\r\n", cmd.filename);
                success = write_on_sd_bit_pack(local_bitmap, cmd.width, cmd.height, cmd.filename);
                break;

            default:
                printf("Core1: Unknown command type %d\r\n", cmd.cmd_type);
                break;
        }

        if (success) {
            printf("Core1: Write completed successfully\r\n");
        } else {
            printf("Core1: Write failed\r\n");
        }

        free(local_bitmap);
    }

    // Unmount on shutdown
    if (fs_mounted) {
        f_unmount(g_drive);
        fs_mounted = false;
        printf("Core1: Filesystem unmounted\r\n");
    }

    core1_running = false;
    printf("Core1: SD writer stopped\r\n");
}

// ============================================================================
// PUBLIC API (Called from Core0)
// ============================================================================

void sd_writer_init(void)
{
    queue_init(&sd_command_queue, sizeof(SD_Write_Command), 4);
    mutex_init(&bitmap_mutex);

    multicore_launch_core1(core1_sd_writer);

    sleep_ms(100);
    printf("Core0: SD writer system initialized\r\n");
}

void sd_writer_shutdown(void)
{
    if (!core1_running) return;

    SD_Write_Command cmd = {
        .cmd_type = SD_CMD_SHUTDOWN
    };

    queue_add_blocking(&sd_command_queue, &cmd);

    while (core1_running) {
        sleep_ms(10);
    }

    printf("Core0: SD writer system shutdown\r\n");
}

bool sd_write_async(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename)
{
    uint32_t bitmap_size = width * height;

    if (bitmap_size > MAX_BITMAP_SIZE) {
        printf("Core0: ERROR - Bitmap too large (%u > %u)\r\n", bitmap_size, MAX_BITMAP_SIZE);
        return false;
    }

    mutex_enter_blocking(&bitmap_mutex);
    memcpy(shared_bitmap_buffer, bitmap, bitmap_size);
    mutex_exit(&bitmap_mutex);

    SD_Write_Command cmd = {
        .cmd_type = SD_CMD_WRITE_TEXT,
        .bitmap = shared_bitmap_buffer,
        .width = width,
        .height = height
    };

    strncpy(cmd.filename, filename, sizeof(cmd.filename) - 1);
    cmd.filename[sizeof(cmd.filename) - 1] = '\0';

    if (!queue_try_add(&sd_command_queue, &cmd)) {
        printf("Core0: WARNING - SD command queue is full\r\n");
        return false;
    }

    printf("Core0: Queued write to '%s'\r\n", filename);
    return true;
}

bool sd_write_async_packed(const uint8_t *bitmap, uint16_t width, uint16_t height, const char *filename)
{
    uint32_t bitmap_size = width * height;

    if (bitmap_size > MAX_BITMAP_SIZE) {
        printf("Core0: ERROR - Bitmap too large (%u > %u)\r\n", bitmap_size, MAX_BITMAP_SIZE);
        return false;
    }

    mutex_enter_blocking(&bitmap_mutex);
    memcpy(shared_bitmap_buffer, bitmap, bitmap_size);
    mutex_exit(&bitmap_mutex);

    SD_Write_Command cmd = {
        .cmd_type = SD_CMD_WRITE_BINARY,
        .bitmap = shared_bitmap_buffer,
        .width = width,
        .height = height
    };

    strncpy(cmd.filename, filename, sizeof(cmd.filename) - 1);
    cmd.filename[sizeof(cmd.filename) - 1] = '\0';

    if (!queue_try_add(&sd_command_queue, &cmd)) {
        printf("Core0: WARNING - SD command queue is full\r\n");
        return false;
    }

    printf("Core0: Queued packed write to '%s'\r\n", filename);
    return true;
}

uint32_t sd_writer_pending_count(void)
{
    return queue_get_level(&sd_command_queue);
}

bool sd_writer_is_busy(void)
{
    return queue_get_level(&sd_command_queue) > 0;
}