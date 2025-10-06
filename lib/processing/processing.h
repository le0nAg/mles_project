#ifndef __PREPROCESSING_H__
#define __PREPROCESSING_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @docs: Crop the image from first top-right 1 and the last bottom-left 1.
 */
void simple_crop(const uint8_t* img, int w, int h, uint8_t* out, int* out_w, int* out_h);

#ifdef __cplusplus
}
#endif

#endif // __PREPROCESSING_H__
