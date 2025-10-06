#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "processing.h"

void simple_crop(const uint8_t* img, int w, int h, uint8_t* out, int* out_w, int* out_h){
    int left = w, right = -1;

    // Find leftmost and rightmost 1s
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (img[y * w + x]) {
                if (x < left) left = x;
                if (x > right) right = x;
            }
        }
    }

    // Safety check
    if (right < left) {
        printf("No 1s found in bitmap.\n");
        return;
    }

    int cropped_width = right - left + 1;

    // Print cropped bitmap
    *out_w = cropped_width;
    *out_h = h;
    for (int i = 0; i < h; i++) {
        for(int j = 0; j < cropped_width; j++){
            out[i * cropped_width + j] = img[i * w + (left + j)];
        }
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < cropped_width; j++) {
            printf("%d ", out[i * cropped_width + j]);
        }
        printf("\n");
    }
}
