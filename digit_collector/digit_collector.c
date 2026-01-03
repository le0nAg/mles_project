#include "LCD_Touch.h"
#include "LCD_Driver.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"
#include "sd_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"

extern LCD_DIS sLCD_DIS;
extern uint8_t id;

// For LCD re-init after SD
#ifndef SCAN_DIR_DFT
#define SCAN_DIR_DFT D2U_L2R
#endif

// Color definitions (RGB565) - fallbacks if not defined in LCD headers
#ifndef GREEN
#define GREEN   0x07E0
#endif
#ifndef BLUE
#define BLUE    0x001F
#endif
#ifndef RED
#define RED     0xF800
#endif

#define BOX_X0          100
#define BOX_Y0          60
#define BOX_W           200
#define BOX_H           200
#define BOX_X1          (BOX_X0 + BOX_W)
#define BOX_Y1          (BOX_Y0 + BOX_H)

#define SAVE_BTN_X      350
#define SAVE_BTN_Y      80
#define SAVE_BTN_W      100
#define SAVE_BTN_H      40

#define CLEAR_BTN_X     350
#define CLEAR_BTN_Y     140
#define CLEAR_BTN_W     100
#define CLEAR_BTN_H     40

#define PREV_BTN_X      350
#define PREV_BTN_Y      200
#define PREV_BTN_W      45
#define PREV_BTN_H      40

#define NEXT_BTN_X      405
#define NEXT_BTN_Y      200
#define NEXT_BTN_W      45
#define NEXT_BTN_H      40

#define BITMAP_SIZE     28

// Debug macro - set to 1 to enable verbose debugging
#define DEBUG_ENABLED   1

#if DEBUG_ENABLED
#define DEBUG_PRINT(fmt, ...) printf("[DBG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

typedef struct {
    uint8_t bitmap[BOX_H][BOX_W];
    uint8_t downsampled[BITMAP_SIZE][BITMAP_SIZE];
    int current_digit;
    int sample_count[10];
    uint16_t last_x, last_y;
    bool was_pressed;
    bool has_drawing;
} CollectorUI;

static CollectorUI g_col;
static bool g_sd_ready = false;
static uint32_t g_last_touch_time = 0;

// Keep-alive timer to prevent screen timeout (if applicable)
static uint32_t g_last_activity_time = 0;
#define ACTIVITY_TIMEOUT_MS 30000  // 30 seconds

static uint16_t TP_Read_ADC(uint8_t CMD) {
    uint16_t Data = 0;
    DEV_Digital_Write(TP_CS_PIN, 0);
    SPI4W_Write_Byte(CMD);
    Driver_Delay_us(200);
    Data = SPI4W_Read_Byte(0x00);
    Data <<= 8;
    Data |= SPI4W_Read_Byte(0x00);
    Data >>= 3;
    DEV_Digital_Write(TP_CS_PIN, 1);
    return Data;
}

static uint16_t TP_Read_ADC_Average(uint8_t CMD) {
    uint8_t i, j;
    uint16_t buf[5], sum = 0, temp;
    spi_set_baudrate(SPI_PORT, 3000000);
    for (i = 0; i < 5; i++) {
        buf[i] = TP_Read_ADC(CMD);
        Driver_Delay_us(200);
    }
    spi_set_baudrate(SPI_PORT, 18000000);
    for (i = 0; i < 4; i++) {
        for (j = i + 1; j < 5; j++) {
            if (buf[i] > buf[j]) { temp = buf[i]; buf[i] = buf[j]; buf[j] = temp; }
        }
    }
    for (i = 1; i < 4; i++) sum += buf[i];
    return sum / 3;
}

static bool TP_Read_Coords(uint16_t *x, uint16_t *y) {
    uint16_t x1, y1, x2, y2;
    x1 = TP_Read_ADC_Average(0xD0);
    y1 = TP_Read_ADC_Average(0x90);
    Driver_Delay_us(100);
    x2 = TP_Read_ADC_Average(0xD0);
    y2 = TP_Read_ADC_Average(0x90);
    
    int dx = (int)x1 - (int)x2;
    int dy = (int)y1 - (int)y2;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    if (dx > 50 || dy > 50) return false;
    
    *x = (x1 + x2) / 2;
    *y = (y1 + y2) / 2;
    return true;
}

static bool TP_Scan_Internal(uint16_t *px, uint16_t *py) {
    if (DEV_Digital_Read(TP_IRQ_PIN) == 0) {
        uint16_t raw_x, raw_y;
        if (!TP_Read_Coords(&raw_x, &raw_y)) return false;
        
        int16_t tx, ty;
        
        if (id == LCD_2_8) {
            tx = (int16_t)(0.066626f * raw_x - 20);
            ty = (int16_t)(0.089779f * raw_y - 34);
        } else {
            tx = (int16_t)(0.132443f * raw_y - 22);
            ty = (int16_t)(-0.089997f * raw_x + 320);
        }
        
        if (tx < 0) tx = 0;
        if (ty < 0) ty = 0;
        if (tx > (int16_t)sLCD_DIS.LCD_Dis_Column) tx = sLCD_DIS.LCD_Dis_Column;
        if (ty > (int16_t)sLCD_DIS.LCD_Dis_Page) ty = sLCD_DIS.LCD_Dis_Page;
        
        *px = (uint16_t)tx;
        *py = (uint16_t)ty;
        return true;
    }
    return false;
}

static void draw_box(void) {
    GUI_DrawRectangle(BOX_X0 + 1, BOX_Y0 + 1, BOX_X1 - 1, BOX_Y1 - 1, 
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(BOX_X0, BOX_Y0, BOX_X1, BOX_Y1, 
                      BLACK, DRAW_EMPTY, DOT_PIXEL_2X2);
}

static void draw_buttons(void) {
    GUI_DrawRectangle(SAVE_BTN_X, SAVE_BTN_Y, SAVE_BTN_X + SAVE_BTN_W, SAVE_BTN_Y + SAVE_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(SAVE_BTN_X, SAVE_BTN_Y, SAVE_BTN_X + SAVE_BTN_W, SAVE_BTN_Y + SAVE_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(SAVE_BTN_X + 25, SAVE_BTN_Y + 12, "SAVE", &Font16, WHITE, BLACK);
    
    GUI_DrawRectangle(CLEAR_BTN_X, CLEAR_BTN_Y, CLEAR_BTN_X + CLEAR_BTN_W, CLEAR_BTN_Y + CLEAR_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(CLEAR_BTN_X, CLEAR_BTN_Y, CLEAR_BTN_X + CLEAR_BTN_W, CLEAR_BTN_Y + CLEAR_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(CLEAR_BTN_X + 20, CLEAR_BTN_Y + 12, "CLEAR", &Font16, WHITE, BLACK);
    
    GUI_DrawRectangle(PREV_BTN_X, PREV_BTN_Y, PREV_BTN_X + PREV_BTN_W, PREV_BTN_Y + PREV_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(PREV_BTN_X, PREV_BTN_Y, PREV_BTN_X + PREV_BTN_W, PREV_BTN_Y + PREV_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(PREV_BTN_X + 15, PREV_BTN_Y + 12, "<", &Font16, WHITE, BLACK);
    
    GUI_DrawRectangle(NEXT_BTN_X, NEXT_BTN_Y, NEXT_BTN_X + NEXT_BTN_W, NEXT_BTN_Y + NEXT_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(NEXT_BTN_X, NEXT_BTN_Y, NEXT_BTN_X + NEXT_BTN_W, NEXT_BTN_Y + NEXT_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(NEXT_BTN_X + 15, NEXT_BTN_Y + 12, ">", &Font16, WHITE, BLACK);
}

static void draw_status(void) {
    GUI_DrawRectangle(350, 10, 470, 70, LCD_BACKGROUND, DRAW_FULL, DOT_PIXEL_1X1);
    
    char buf[32];
    snprintf(buf, sizeof(buf), "Digit: %d", g_col.current_digit);
    GUI_DisString_EN(350, 15, buf, &Font20, LCD_BACKGROUND, BLACK);
    
    snprintf(buf, sizeof(buf), "Saved: %d/10", g_col.sample_count[g_col.current_digit]);
    GUI_DisString_EN(350, 45, buf, &Font16, LCD_BACKGROUND, BLACK);
}

static void draw_total_status(void) {
    GUI_DrawRectangle(10, 270, 300, 310, LCD_BACKGROUND, DRAW_FULL, DOT_PIXEL_1X1);
    
    char buf[64];
    snprintf(buf, sizeof(buf), "0:%d 1:%d 2:%d 3:%d 4:%d", 
             g_col.sample_count[0], g_col.sample_count[1], g_col.sample_count[2],
             g_col.sample_count[3], g_col.sample_count[4]);
    GUI_DisString_EN(10, 275, buf, &Font12, LCD_BACKGROUND, BLACK);
    
    snprintf(buf, sizeof(buf), "5:%d 6:%d 7:%d 8:%d 9:%d", 
             g_col.sample_count[5], g_col.sample_count[6], g_col.sample_count[7],
             g_col.sample_count[8], g_col.sample_count[9]);
    GUI_DisString_EN(10, 290, buf, &Font12, LCD_BACKGROUND, BLACK);
}

static void draw_sd_status(const char *msg, uint16_t color) {
    GUI_DrawRectangle(350, 245, 470, 270, LCD_BACKGROUND, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(350, 250, msg, &Font12, LCD_BACKGROUND, color);
}

static void draw_full_ui(void) {
    DEBUG_PRINT("Drawing full UI...");
    LCD_Clear(LCD_BACKGROUND);
    GUI_DisString_EN(10, 10, "Digit Data Collection", &Font16, LCD_BACKGROUND, BLACK);
    GUI_DisString_EN(10, 35, "Draw digit, press SAVE", &Font12, LCD_BACKGROUND, BLACK);
    draw_box();
    draw_buttons();
    draw_status();
    draw_total_status();
    DEBUG_PRINT("Full UI drawn");
}

static void restore_spi_for_lcd(void) {
    // Restore SPI settings for LCD after SD operations
    DEBUG_PRINT("Restoring SPI settings for LCD");
    spi_set_baudrate(SPI_PORT, 18000000);
    
    // Make sure LCD CS is properly configured
    DEV_Digital_Write(LCD_CS_PIN, 1);
    DEV_Digital_Write(TP_CS_PIN, 1);
    
    // Small delay to let things settle
    sleep_ms(10);
}

static void crop_and_downsample(void) {
    int min_x = BOX_W, max_x = 0, min_y = BOX_H, max_y = 0;
    
    for (int y = 0; y < BOX_H; y++) {
        for (int x = 0; x < BOX_W; x++) {
            if (g_col.bitmap[y][x]) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    
    if (min_x > max_x || min_y > max_y) {
        memset(g_col.downsampled, 0, sizeof(g_col.downsampled));
        return;
    }
    
    int crop_w = max_x - min_x + 1;
    int crop_h = max_y - min_y + 1;
    
    int pad = 2;
    int target_size = BITMAP_SIZE - 2 * pad;
    
    float scale;
    int offset_x, offset_y;
    if (crop_w > crop_h) {
        scale = (float)target_size / crop_w;
        offset_x = pad;
        offset_y = pad + (target_size - (int)(crop_h * scale)) / 2;
    } else {
        scale = (float)target_size / crop_h;
        offset_x = pad + (target_size - (int)(crop_w * scale)) / 2;
        offset_y = pad;
    }
    
    memset(g_col.downsampled, 0, sizeof(g_col.downsampled));
    
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            if (g_col.bitmap[y][x]) {
                int dst_x = offset_x + (int)((x - min_x) * scale);
                int dst_y = offset_y + (int)((y - min_y) * scale);
                if (dst_x >= 0 && dst_x < BITMAP_SIZE && dst_y >= 0 && dst_y < BITMAP_SIZE) {
                    g_col.downsampled[dst_y][dst_x] = 1;
                }
            }
        }
    }
}

static void save_sample(void) {
    if (!g_sd_ready) {
        printf("SD card not ready, cannot save\n");
        draw_sd_status("SD: Error!", RED);
        return;
    }
    
    if (!g_col.has_drawing) {
        printf("No drawing to save\n");
        return;
    }
    
    DEBUG_PRINT("Waiting for SD writer...");
    
    // Wait if SD writer is busy from previous write (with timeout)
    int timeout = 100;  // 1 second timeout
    while (sd_writer_is_busy() && timeout > 0) {
        sleep_ms(10);
        timeout--;
    }
    
    if (timeout == 0) {
        printf("SD writer timeout!\n");
        draw_sd_status("SD: Timeout!", RED);
        return;
    }
    
    crop_and_downsample();
    
    // Filename encodes the label: digit_X_NN.txt where X is the digit
    char filename[64];
    snprintf(filename, sizeof(filename), "digit_%d_%02d.txt", 
             g_col.current_digit, g_col.sample_count[g_col.current_digit]);
    
    // Write bitmap as ASCII (rows of 0s and 1s)
    uint8_t flat_bitmap[BITMAP_SIZE * BITMAP_SIZE];
    for (int y = 0; y < BITMAP_SIZE; y++) {
        for (int x = 0; x < BITMAP_SIZE; x++) {
            flat_bitmap[y * BITMAP_SIZE + x] = g_col.downsampled[y][x];
        }
    }
    
    DEBUG_PRINT("Writing to SD: %s", filename);
    
    if (!sd_write_async(flat_bitmap, BITMAP_SIZE, BITMAP_SIZE, filename)) {
        printf("Failed to queue bitmap write: %s\n", filename);
        draw_sd_status("Write fail!", RED);
        // Restore SPI for LCD after SD failure
        restore_spi_for_lcd();
        return;
    }
    
    // Restore SPI settings for LCD after SD write
    restore_spi_for_lcd();
    
    g_col.sample_count[g_col.current_digit]++;
    printf("Saved: %s (%d samples for digit %d)\n", 
           filename, g_col.sample_count[g_col.current_digit], g_col.current_digit);
    
    draw_status();
    draw_total_status();
    draw_sd_status("SD: OK", GREEN);
    
    memset(g_col.bitmap, 0, sizeof(g_col.bitmap));
    g_col.has_drawing = false;
    draw_box();
}

static void clear_drawing(void) {
    memset(g_col.bitmap, 0, sizeof(g_col.bitmap));
    g_col.has_drawing = false;
    g_col.last_x = 0xFFFF;
    g_col.last_y = 0xFFFF;
    draw_box();
}

static void change_digit(int delta) {
    g_col.current_digit += delta;
    if (g_col.current_digit < 0) g_col.current_digit = 9;
    if (g_col.current_digit > 9) g_col.current_digit = 0;
    clear_drawing();
    draw_status();
}

static void draw_pixel_at(uint16_t x, uint16_t y) {
    if (x <= BOX_X0 + 2 || x >= BOX_X1 - 2 || y <= BOX_Y0 + 2 || y >= BOX_Y1 - 2) return;
    
    int bx = x - BOX_X0;
    int by = y - BOX_Y0;
    
    if (bx >= 0 && bx < BOX_W && by >= 0 && by < BOX_H) {
        g_col.bitmap[by][bx] = 1;
        if (bx > 0) g_col.bitmap[by][bx-1] = 1;
        if (bx < BOX_W-1) g_col.bitmap[by][bx+1] = 1;
        if (by > 0) g_col.bitmap[by-1][bx] = 1;
        if (by < BOX_H-1) g_col.bitmap[by+1][bx] = 1;
        g_col.has_drawing = true;
    }
    
    LCD_SetPointlColor(x, y, BLACK);
    LCD_SetPointlColor(x+1, y, BLACK);
    LCD_SetPointlColor(x, y+1, BLACK);
    LCD_SetPointlColor(x+1, y+1, BLACK);
}

static void draw_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    
    int steps = (dx > dy) ? dx : dy;
    if (steps == 0) {
        draw_pixel_at((uint16_t)x0, (uint16_t)y0);
        return;
    }
    
    float x_inc = (float)(x1 - x0) / steps;
    float y_inc = (float)(y1 - y0) / steps;
    float x = (float)x0, y = (float)y0;
    
    for (int i = 0; i <= steps; i++) {
        draw_pixel_at((uint16_t)(x + 0.5f), (uint16_t)(y + 0.5f));
        x += x_inc;
        y += y_inc;
    }
}

static bool in_box(uint16_t x, uint16_t y) {
    return (x > BOX_X0 && x < BOX_X1 && y > BOX_Y0 && y < BOX_Y1);
}

static bool check_button(uint16_t x, uint16_t y, uint16_t bx, uint16_t by, uint16_t bw, uint16_t bh) {
    return (x >= bx && x <= bx + bw && y >= by && y <= by + bh);
}

void Collector_Init(void) {
    DEBUG_PRINT("Collector_Init called");
    memset(&g_col, 0, sizeof(g_col));
    g_col.last_x = 0xFFFF;
    g_col.last_y = 0xFFFF;
    g_sd_ready = false;
    g_last_activity_time = to_ms_since_boot(get_absolute_time());
    DEBUG_PRINT("Collector_Init complete");
}

void Collector_DrawScreen(void) {
    DEBUG_PRINT("=== Collector_DrawScreen START ===");
    
    // Draw initial UI first so user sees something immediately
    draw_full_ui();
    draw_sd_status("SD: Init...", BLUE);
    
    DEBUG_PRINT("Initial UI drawn, now initializing SD...");
    printf("Initializing SD card writer...\n");
    
    // Initialize SD with timeout protection
    uint32_t sd_start = to_ms_since_boot(get_absolute_time());
    
    // Try SD init - if it hangs, the watchdog should catch it
    // Note: You may need to implement a timeout in sd_writer_init()
    sd_writer_init();
    
    uint32_t sd_elapsed = to_ms_since_boot(get_absolute_time()) - sd_start;
    DEBUG_PRINT("SD init took %lu ms", sd_elapsed);
    
    // Give SD some time to mount (reduced from 1000ms)
    sleep_ms(500);
    DEBUG_PRINT("Post-SD delay complete");
    
    // CRITICAL: Restore SPI settings for LCD before re-init
    DEBUG_PRINT("Restoring SPI for LCD...");
    restore_spi_for_lcd();
    
    // Re-initialize LCD after SD init
    DEBUG_PRINT("Re-initializing LCD...");
    LCD_SCAN_DIR lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir, 1000);
    DEBUG_PRINT("LCD re-init complete");
    
    // Small delay to let LCD stabilize
    sleep_ms(100);
    
    // Redraw everything
    DEBUG_PRINT("Redrawing UI after LCD re-init...");
    draw_full_ui();
    
    g_sd_ready = true;
    draw_sd_status("SD: OK", GREEN);
    
    g_last_activity_time = to_ms_since_boot(get_absolute_time());
    
    DEBUG_PRINT("=== Collector_DrawScreen COMPLETE ===");
    printf("SD init complete, system ready\n");
}

// Alternative initialization without SD - for testing
void Collector_DrawScreen_NoSD(void) {
    DEBUG_PRINT("=== Collector_DrawScreen_NoSD START ===");
    
    draw_full_ui();
    
    g_sd_ready = false;
    draw_sd_status("SD: Disabled", BLUE);
    
    g_last_activity_time = to_ms_since_boot(get_absolute_time());
    
    DEBUG_PRINT("=== Collector_DrawScreen_NoSD COMPLETE ===");
    printf("System ready (SD disabled for testing)\n");
}

void Collector_Run(void) {
    uint16_t x, y;
    bool pressed = TP_Scan_Internal(&x, &y);
    
    if (pressed) {
        // Update activity time on any touch
        g_last_activity_time = to_ms_since_boot(get_absolute_time());
        
        if (in_box(x, y)) {
            if (g_col.last_x != 0xFFFF && g_col.last_y != 0xFFFF) {
                draw_line((int16_t)g_col.last_x, (int16_t)g_col.last_y, (int16_t)x, (int16_t)y);
            } else {
                draw_pixel_at(x, y);
            }
            g_col.last_x = x;
            g_col.last_y = y;
        }
        
        if (!g_col.was_pressed) {
            if (check_button(x, y, SAVE_BTN_X, SAVE_BTN_Y, SAVE_BTN_W, SAVE_BTN_H)) {
                DEBUG_PRINT("SAVE button pressed");
                save_sample();
            } else if (check_button(x, y, CLEAR_BTN_X, CLEAR_BTN_Y, CLEAR_BTN_W, CLEAR_BTN_H)) {
                DEBUG_PRINT("CLEAR button pressed");
                clear_drawing();
            } else if (check_button(x, y, PREV_BTN_X, PREV_BTN_Y, PREV_BTN_W, PREV_BTN_H)) {
                DEBUG_PRINT("PREV button pressed");
                change_digit(-1);
            } else if (check_button(x, y, NEXT_BTN_X, NEXT_BTN_Y, NEXT_BTN_W, NEXT_BTN_H)) {
                DEBUG_PRINT("NEXT button pressed");
                change_digit(1);
            }
        }
        
        g_col.was_pressed = true;
    } else {
        g_col.was_pressed = false;
        g_col.last_x = 0xFFFF;
        g_col.last_y = 0xFFFF;
    }
}

// Call this periodically to keep the display alive (if needed)
void Collector_KeepAlive(void) {
    uint32_t now = to_ms_since_boot(get_absolute_time());
    
    // If no activity for a while, do a minimal LCD operation to keep it alive
    if (now - g_last_activity_time > ACTIVITY_TIMEOUT_MS) {
        DEBUG_PRINT("Keep-alive triggered");
        // Just update a small area - this may help prevent sleep/timeout
        draw_sd_status(g_sd_ready ? "SD: OK" : "SD: N/A", g_sd_ready ? GREEN : BLUE);
        g_last_activity_time = now;
    }
}

// Get SD status for external checking
bool Collector_IsSDReady(void) {
    return g_sd_ready;
}