/**
 * @file digit_preprocess.h
 * @brief Preprocessing for hand-drawn digits to match MNIST format
 */

#ifndef DIGIT_PREPROCESS_H
#define DIGIT_PREPROCESS_H

#include <stdint.h>
#include <string.h>

/**
 * @brief Find bounding box of drawn content
 */
static inline void find_bounding_box(const uint8_t bitmap[28][28], 
                                      int* min_x, int* min_y, 
                                      int* max_x, int* max_y) {
    *min_x = 28; *min_y = 28;
    *max_x = 0;  *max_y = 0;
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (bitmap[y][x]) {
                if (x < *min_x) *min_x = x;
                if (x > *max_x) *max_x = x;
                if (y < *min_y) *min_y = y;
                if (y > *max_y) *max_y = y;
            }
        }
    }
}

/**
 * @brief Center the digit in a 20x20 area with 4px padding (MNIST style)
 */
static inline void center_digit(const uint8_t src[28][28], uint8_t dst[28][28]) {
    memset(dst, 0, 28 * 28);
    
    int min_x, min_y, max_x, max_y;
    find_bounding_box(src, &min_x, &min_y, &max_x, &max_y);
    
    if (max_x < min_x || max_y < min_y) return;  // Empty
    
    int src_w = max_x - min_x + 1;
    int src_h = max_y - min_y + 1;
    
    // Target: 20x20 content area centered in 28x28 (4px padding)
    int target_size = 20;
    float scale = (float)target_size / (src_w > src_h ? src_w : src_h);
    
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    
    // Center in 28x28
    int offset_x = (28 - new_w) / 2;
    int offset_y = (28 - new_h) / 2;
    
    // Simple nearest-neighbor scaling and centering
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = min_x + (x * src_w) / new_w;
            int src_y = min_y + (y * src_h) / new_h;
            
            if (src_x >= 0 && src_x < 28 && src_y >= 0 && src_y < 28) {
                dst[offset_y + y][offset_x + x] = src[src_y][src_x];
            }
        }
    }
}

/**
 * @brief Thicken strokes using dilation (3x3)
 */
static inline void thicken_strokes(const uint8_t src[28][28], uint8_t dst[28][28]) {
    memset(dst, 0, 28 * 28);
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (src[y][x]) {
                // 3x3 dilation
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                            dst[ny][nx] = 1;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Full preprocessing pipeline: thicken -> center
 */
static inline void preprocess_digit(const uint8_t raw[28][28], uint8_t processed[28][28]) {
    uint8_t temp[28][28];
    
    // Step 1: Thicken strokes
    thicken_strokes(raw, temp);
    
    // Step 2: Center the digit
    center_digit(temp, processed);
}

/**
 * @brief Convert preprocessed bitmap to model input format
 * @param bitmap Preprocessed 28x28 bitmap (0/1 values)
 * @param output 784-byte array for model (0-255 values)
 */
static inline void bitmap_to_model_input(const uint8_t bitmap[28][28], uint8_t* output) {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // MNIST: white digit (255) on black background (0)
            output[y * 28 + x] = bitmap[y][x] ? 255 : 0;
        }
    }
}

#endif // DIGIT_PREPROCESS_H