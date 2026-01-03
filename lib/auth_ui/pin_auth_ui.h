#ifndef PIN_AUTH_UI_H
#define PIN_AUTH_UI_H

#include <stdint.h>
#include <stdbool.h>

#define PIN_DIGIT_COUNT 3
#define PIN_BITMAP_SIZE 28

void PinAuth_Init(void);
void PinAuth_DrawScreen(void);
void PinAuth_HandleTouch(uint16_t x, uint16_t y, bool pressed);
void PinAuth_Run(void);
const uint8_t* PinAuth_GetDigitBitmap(int idx);

#endif