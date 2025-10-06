#ifndef __PREPROCESSING_H__
#define __PREPROCESSING_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @docs: Crop the image from first top-left 1 and the last bottom-right 1.
 */
void simple_crop(const uint8_t* img, int w, int h, uint8_t* out, int* out_w, int* out_h);

/**
 * @docs: Find connected components in a binary image.
 */
void connected_components(const uint8_t* img, int w, int h, int* n_components);
static void explode_vertex(const uint8_t* img, int w, int h, uint8_t* out, int16_t magnitude);

/**
 * @docs: Perform edge detection on an image.
 * @todo: Implement this function.
 * considerations: could use Sobel operator or Morphological detection.
 */
void edge_detection(const uint8_t* img, int w, int h, uint8_t* out);
static void morphological_explode(const uint8_t* img, int w, int h, uint8_t* out);
static void morphological_implode(const uint8_t* img, int w, int h, uint8_t* out);
/**
 * @docs: Compute the density of an image.
 * @todo: Implement optimization.
 * @optimization: current implementation is alone, make sense in the future to combine with other functions
 *  to avoid double matrix iteration.
 */
void density(const uint8_t* img, int w, int h, float* density);

/**
 * !! should be Topology-Preserving
 * https://arxiv.org/abs/2407.17786
 * @todo    
 */
void downsample(const uint8_t* img, int w, int h, uint8_t* out, int factor);

// compactness feats
static int area(const uint8_t* img, int w, int h);
static int perimeter(const uint8_t* img, int w, int h);
void compactness(const uint8_t* img, int w, int h, float* isoperimetric, float* a_to_p_ratio, float* circularity_ratio);

#ifdef __cplusplus
}
#endif

#endif // __PREPROCESSING_H__
