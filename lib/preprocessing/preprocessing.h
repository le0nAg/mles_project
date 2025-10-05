#ifndef __PREPROCESSING_H__
#define __PREPROCESSING_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Preprocessing function declarations
/*
    @docs: implemented with sobel operator
*/ 
void corner_detection(const uint8_t* input_image, uint8_t* output_image, int width, int height);
void edge_detection(const uint8_t* input_image, uint8_t* output_image, int width, int height);
void downsample(const uint8_t* input_image, uint8_t* output_image, int width, int height, int factor);
void crop_image(const uint8_t* input_image, uint8_t* output_image, int width, int height, int x, int y, int crop_width, int crop_height);


#ifdef __cplusplus
}
#endif

#endif // __PREPROCESSING_H__
