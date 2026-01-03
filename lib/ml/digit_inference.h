#ifndef DIGIT_INFERENCE_H
#define DIGIT_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DIGIT_IMG_WIDTH  28
#define DIGIT_IMG_HEIGHT 28
#define DIGIT_IMG_SIZE   (DIGIT_IMG_WIDTH * DIGIT_IMG_HEIGHT)

#define DIGIT_NUM_CLASSES 10

#define DIGIT_OK              0
#define DIGIT_ERR_INIT       -1
#define DIGIT_ERR_INVOKE     -2
#define DIGIT_ERR_NULL_INPUT -3

int digit_inference_init(void);

int digit_inference_predict(const uint8_t* image_28x28, uint8_t* confidence);

int digit_inference_get_scores(const uint8_t* image_28x28, int8_t* scores);

bool digit_inference_is_ready(void);

void digit_inference_free(void);

const char* digit_inference_get_info(void);

#ifdef __cplusplus
}
#endif

#endif
