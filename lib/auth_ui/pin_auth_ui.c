#include "LCD_Touch.h"
#include "LCD_Driver.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"
#include "digit_inference.h"  // <-- Add ML inference
#include "digit_preprocess.h" // <-- Add preprocessing
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"

extern LCD_DIS sLCD_DIS;
extern uint8_t id;

#define DIGIT_BOX_W     80
#define DIGIT_BOX_H     100
#define DIGIT_BOX_GAP   20
#define DIGIT_BOX_Y     80
#define NUM_DIGITS      3

#define AUTH_BTN_X      400
#define AUTH_BTN_Y      10
#define AUTH_BTN_W      70
#define AUTH_BTN_H      30

#define CLEAR_BTN_X     10
#define CLEAR_BTN_Y     10
#define CLEAR_BTN_W     70
#define CLEAR_BTN_H     30

#define RESULT_X        380
#define RESULT_Y        100

// Confidence display area
#define CONF_X          320
#define CONF_Y          200

static const char* HARDCODED_PIN = "021";

typedef struct {
    uint16_t x0, y0, x1, y1;
    uint8_t bitmap[28][28];
    int8_t recognized_digit;
    uint8_t confidence;      // <-- Add confidence
    bool has_drawing;
} DigitBox;

typedef struct {
    DigitBox boxes[NUM_DIGITS];
    int active_box;
    int auth_state;
    char entered_pin[NUM_DIGITS + 1];
    uint16_t last_x, last_y;
    bool was_pressed;
    bool ml_initialized;     // <-- Track ML init state
} PinAuthUI;

static PinAuthUI g_ui;

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

static uint16_t calc_digit_box_x(int idx) {
    uint16_t start_x = 30;
    return start_x + idx * (DIGIT_BOX_W + DIGIT_BOX_GAP);
}

static void draw_digit_box_frame(int idx, bool selected) {
    DigitBox* box = &g_ui.boxes[idx];
    GUI_DrawRectangle(box->x0, box->y0, box->x1, box->y1, 
                      BLACK, DRAW_EMPTY, selected ? DOT_PIXEL_2X2 : DOT_PIXEL_1X1);
    
    char label[4];
    snprintf(label, sizeof(label), "%d", idx + 1);
    GUI_DisString_EN(box->x0 + 35, box->y0 - 18, label, &Font16, LCD_BACKGROUND, BLACK);
}

static void draw_digit_box_full(int idx, bool selected) {
    DigitBox* box = &g_ui.boxes[idx];
    
    GUI_DrawRectangle(box->x0 + 1, box->y0 + 1, box->x1 - 1, box->y1 - 1, 
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    draw_digit_box_frame(idx, selected);
    
    if (box->recognized_digit >= 0) {
        char digit_str[2] = {(char)('0' + box->recognized_digit), '\0'};
        GUI_DisString_EN(box->x0 + 30, box->y1 + 5, digit_str, &Font20, LCD_BACKGROUND, BLACK);
    }
}

static void draw_buttons(void) {
    GUI_DrawRectangle(CLEAR_BTN_X, CLEAR_BTN_Y, 
                      CLEAR_BTN_X + CLEAR_BTN_W, CLEAR_BTN_Y + CLEAR_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(CLEAR_BTN_X, CLEAR_BTN_Y, 
                      CLEAR_BTN_X + CLEAR_BTN_W, CLEAR_BTN_Y + CLEAR_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(CLEAR_BTN_X + 8, CLEAR_BTN_Y + 8, "CLEAR", &Font16, WHITE, BLACK);
    
    GUI_DrawRectangle(AUTH_BTN_X, AUTH_BTN_Y,
                      AUTH_BTN_X + AUTH_BTN_W, AUTH_BTN_Y + AUTH_BTN_H,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(AUTH_BTN_X, AUTH_BTN_Y,
                      AUTH_BTN_X + AUTH_BTN_W, AUTH_BTN_Y + AUTH_BTN_H,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_1X1);
    GUI_DisString_EN(AUTH_BTN_X + 12, AUTH_BTN_Y + 8, "AUTH", &Font16, WHITE, BLACK);
}

static void draw_result(void) {
    GUI_DrawRectangle(RESULT_X, RESULT_Y, RESULT_X + 90, RESULT_Y + 60,
                      LCD_BACKGROUND, DRAW_FULL, DOT_PIXEL_1X1);
    
    if (g_ui.auth_state == 1) {
        GUI_DisString_EN(RESULT_X, RESULT_Y + 10, "ACCESS", &Font16, LCD_BACKGROUND, BLACK);
        GUI_DisString_EN(RESULT_X, RESULT_Y + 30, "GRANTED", &Font16, LCD_BACKGROUND, BLACK);
    } else if (g_ui.auth_state == 2) {
        GUI_DisString_EN(RESULT_X, RESULT_Y + 10, "ACCESS", &Font16, LCD_BACKGROUND, BLACK);
        GUI_DisString_EN(RESULT_X, RESULT_Y + 30, "DENIED", &Font16, LCD_BACKGROUND, BLACK);
    }
}

// Draw confidence values for each digit
static void draw_confidence(void) {
    // Clear confidence area
    GUI_DrawRectangle(CONF_X, CONF_Y, CONF_X + 150, CONF_Y + 80,
                      LCD_BACKGROUND, DRAW_FULL, DOT_PIXEL_1X1);
    
    GUI_DisString_EN(CONF_X, CONF_Y, "Confidence:", &Font12, LCD_BACKGROUND, BLACK);
    
    for (int i = 0; i < NUM_DIGITS; i++) {
        char conf_str[16];
        if (g_ui.boxes[i].recognized_digit >= 0) {
            int conf_percent = (g_ui.boxes[i].confidence * 100) / 255;
            snprintf(conf_str, sizeof(conf_str), "D%d: %d%% (%d)", 
                     i + 1, conf_percent, g_ui.boxes[i].recognized_digit);
        } else {
            snprintf(conf_str, sizeof(conf_str), "D%d: --", i + 1);
        }
        GUI_DisString_EN(CONF_X, CONF_Y + 15 + i * 15, conf_str, &Font12, LCD_BACKGROUND, BLACK);
    }
}

static int check_which_box(uint16_t x, uint16_t y) {
    for (int i = 0; i < NUM_DIGITS; i++) {
        DigitBox* box = &g_ui.boxes[i];
        if (x >= box->x0 && x <= box->x1 && y >= box->y0 && y <= box->y1) {
            return i;
        }
    }
    return -1;
}

static void draw_pixel_at(uint16_t x, uint16_t y) {
    int box_idx = check_which_box(x, y);
    if (box_idx < 0) return;
    
    DigitBox* box = &g_ui.boxes[box_idx];
    
    if (x <= box->x0 + 2 || x >= box->x1 - 2 || 
        y <= box->y0 + 2 || y >= box->y1 - 2) return;
    
    int local_x = x - box->x0 - 2;
    int local_y = y - box->y0 - 2;
    int inner_w = DIGIT_BOX_W - 4;
    int inner_h = DIGIT_BOX_H - 4;
    int bmp_x = (local_x * 28) / inner_w;
    int bmp_y = (local_y * 28) / inner_h;
    
    if (bmp_x >= 0 && bmp_x < 28 && bmp_y >= 0 && bmp_y < 28) {
        box->bitmap[bmp_y][bmp_x] = 1;
        box->has_drawing = true;
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

// Convert bitmap (0/1) to model input format (0-255, inverted if needed)
static void prepare_model_input(const uint8_t bitmap[28][28], uint8_t* output) {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Model expects white digit on black background (like MNIST)
            // bitmap: 1 = drawn (black), 0 = empty (white)
            // Convert: drawn -> 255, empty -> 0
            output[y * 28 + x] = bitmap[y][x] ? 255 : 0;
        }
    }
}

// Recognize digit using ML model
static int8_t recognize_digit_ml(const uint8_t bitmap[28][28], uint8_t* confidence) {
    if (!g_ui.ml_initialized) {
        printf("ML not initialized!\n");
        *confidence = 0;
        return -1;
    }
    
    // Check if there's any drawing
    int count = 0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (bitmap[y][x]) count++;
        }
    }
    if (count < 5) {
        *confidence = 0;
        return -1;  // Not enough pixels
    }
    
    // Preprocess: thicken and center (MNIST style)
    uint8_t processed[28][28];
    preprocess_digit(bitmap, processed);
    
    // Convert to model input format
    uint8_t model_input[28 * 28];
    bitmap_to_model_input(processed, model_input);
    
    // Debug: print processed bitmap
    printf("Processed bitmap:\n");
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            printf("%c", processed[y][x] ? '#' : '.');
        }
        printf("\n");
    }
    
    // Run inference
    int result = digit_inference_predict(model_input, confidence);
    
    if (result < 0) {
        printf("Inference error: %d\n", result);
        *confidence = 0;
        return -1;
    }
    
    return (int8_t)result;
}

static void run_authentication(void) {
    printf("=== AUTHENTICATION ===\n");
    
    // Recognize each digit using ML
    for (int i = 0; i < NUM_DIGITS; i++) {
        if (g_ui.boxes[i].has_drawing) {
            g_ui.boxes[i].recognized_digit = recognize_digit_ml(
                g_ui.boxes[i].bitmap, 
                &g_ui.boxes[i].confidence
            );
            printf("Box %d: recognized %d (conf: %d%%)\n", 
                   i, g_ui.boxes[i].recognized_digit,
                   (g_ui.boxes[i].confidence * 100) / 255);
        } else {
            g_ui.boxes[i].recognized_digit = -1;
            g_ui.boxes[i].confidence = 0;
            printf("Box %d: no drawing\n", i);
        }
        
        g_ui.entered_pin[i] = (g_ui.boxes[i].recognized_digit >= 0) 
            ? '0' + g_ui.boxes[i].recognized_digit : '?';
    }
    g_ui.entered_pin[NUM_DIGITS] = '\0';
    
    printf("PIN: %s (expected: %s)\n", g_ui.entered_pin, HARDCODED_PIN);
    
    // Check PIN
    g_ui.auth_state = (strcmp(g_ui.entered_pin, HARDCODED_PIN) == 0) ? 1 : 2;
    
    // Update display
    draw_result();
    draw_confidence();
    
    // Show recognized digits below boxes
    for (int i = 0; i < NUM_DIGITS; i++) {
        if (g_ui.boxes[i].recognized_digit >= 0) {
            char digit_str[2] = {(char)('0' + g_ui.boxes[i].recognized_digit), '\0'};
            GUI_DisString_EN(g_ui.boxes[i].x0 + 30, g_ui.boxes[i].y1 + 5, 
                            digit_str, &Font20, LCD_BACKGROUND, BLACK);
        }
    }
}

static bool check_button(uint16_t x, uint16_t y, uint16_t bx, uint16_t by, uint16_t bw, uint16_t bh) {
    return (x >= bx && x <= bx + bw && y >= by && y <= by + bh);
}

void PinAuth_Init(void) {
    memset(&g_ui, 0, sizeof(g_ui));
    
    // Initialize ML model
    int ml_result = digit_inference_init();
    if (ml_result == DIGIT_OK) {
        g_ui.ml_initialized = true;
        printf("ML model initialized: %s\n", digit_inference_get_info());
    } else {
        g_ui.ml_initialized = false;
        printf("ERROR: ML init failed: %d\n", ml_result);
    }
    
    for (int i = 0; i < NUM_DIGITS; i++) {
        g_ui.boxes[i].x0 = calc_digit_box_x(i);
        g_ui.boxes[i].y0 = DIGIT_BOX_Y;
        g_ui.boxes[i].x1 = g_ui.boxes[i].x0 + DIGIT_BOX_W;
        g_ui.boxes[i].y1 = g_ui.boxes[i].y0 + DIGIT_BOX_H;
        g_ui.boxes[i].recognized_digit = -1;
        g_ui.boxes[i].confidence = 0;
        printf("Box %d: x0=%d y0=%d x1=%d y1=%d\n", i, 
               g_ui.boxes[i].x0, g_ui.boxes[i].y0, 
               g_ui.boxes[i].x1, g_ui.boxes[i].y1);
    }
    g_ui.last_x = 0xFFFF;
    g_ui.last_y = 0xFFFF;
}

void PinAuth_DrawScreen(void) {
    LCD_Clear(LCD_BACKGROUND);
    GUI_DisString_EN(140, 10, "PIN Authentication", &Font16, LCD_BACKGROUND, BLACK);
    
    // Show ML status
    if (g_ui.ml_initialized) {
        GUI_DisString_EN(140, 30, "ML Ready", &Font12, LCD_BACKGROUND, BLACK);
    } else {
        GUI_DisString_EN(140, 30, "ML ERROR", &Font12, LCD_BACKGROUND, RED);
    }
    
    draw_buttons();
    for (int i = 0; i < NUM_DIGITS; i++) {
        draw_digit_box_full(i, i == g_ui.active_box);
    }
}

static void clear_all(void) {
    for (int i = 0; i < NUM_DIGITS; i++) {
        memset(g_ui.boxes[i].bitmap, 0, sizeof(g_ui.boxes[i].bitmap));
        g_ui.boxes[i].recognized_digit = -1;
        g_ui.boxes[i].confidence = 0;
        g_ui.boxes[i].has_drawing = false;
    }
    g_ui.active_box = 0;
    g_ui.auth_state = 0;
    memset(g_ui.entered_pin, 0, sizeof(g_ui.entered_pin));
    g_ui.last_x = 0xFFFF;
    g_ui.last_y = 0xFFFF;
    PinAuth_DrawScreen();
}

void PinAuth_Run(void) {
    uint16_t x, y;
    bool pressed = TP_Scan_Internal(&x, &y);
    
    if (pressed) {
        int box = check_which_box(x, y);
        
        if (box >= 0) {
            if (g_ui.last_x != 0xFFFF && g_ui.last_y != 0xFFFF) {
                draw_line((int16_t)g_ui.last_x, (int16_t)g_ui.last_y, (int16_t)x, (int16_t)y);
            } else {
                draw_pixel_at(x, y);
            }
            g_ui.last_x = x;
            g_ui.last_y = y;
        }
        
        if (!g_ui.was_pressed) {
            if (check_button(x, y, CLEAR_BTN_X, CLEAR_BTN_Y, CLEAR_BTN_W, CLEAR_BTN_H)) {
                clear_all();
            } else if (check_button(x, y, AUTH_BTN_X, AUTH_BTN_Y, AUTH_BTN_W, AUTH_BTN_H)) {
                run_authentication();
            }
        }
        
        g_ui.was_pressed = true;
    } else {
        g_ui.was_pressed = false;
        g_ui.last_x = 0xFFFF;
        g_ui.last_y = 0xFFFF;
    }
}

const uint8_t* PinAuth_GetDigitBitmap(int idx) {
    return (idx >= 0 && idx < NUM_DIGITS) ? &g_ui.boxes[idx].bitmap[0][0] : NULL;
}