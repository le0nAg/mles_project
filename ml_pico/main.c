// main.c
// template for final application

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "hardware/irq.h"
#include "hardware/gpio.h"

// Include your inference API
bool init_inference();
bool run_inference_from_buffer(const void *input_buf, size_t input_size);
int get_last_prediction();
float get_last_score();

#include "lib/LCD_Touch.h"
#include "lib/LCD_GUI.h"
#include "lib/LCD_Driver.h"

#define MODEL_INPUT_WIDTH 28
#define MODEL_INPUT_HEIGHT 28
#define MODEL_INPUT_CHANNELS 1

#define MODEL_INPUT_SIZE (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS)

static uint8_t model_input_buffer[MODEL_INPUT_SIZE];

static bool capture_touchscreen_image(uint8_t *out_buf, size_t out_size)
{

    if (out_size < MODEL_INPUT_SIZE) return false;

    memset(out_buf, 0, out_size);

    return true;
}

int main()
{
    stdio_init_all();
    printf("Pico TFLite Micro demo starting...\n");

    // Initialize LCD & touch
    DEV_Module_Init();       
    LCD_Init();               
    LCD_Clear(0xFFFF);        

    if (!init_inference()) {
        printf("Failed to initialize TFLite interpreter\n");
        while (1) { sleep_ms(1000); }
    }

    GUI_Clear(0xFFFF);
    GUI_DrawString_EN(10, 10, "Draw digit, then press BTN0 (GP0) to infer", &Font20, 0, 0);

    const uint PIN_BUTTON = 0;
    gpio_init(PIN_BUTTON);
    gpio_set_dir(PIN_BUTTON, GPIO_IN);
    gpio_pull_up(PIN_BUTTON);

    while (1) {
        if (!gpio_get(PIN_BUTTON)) {
            sleep_ms(50);
            if (!gpio_get(PIN_BUTTON)) {
                printf("Button pressed — capturing input and running inference\n");

                if (!capture_touchscreen_image(model_input_buffer, sizeof(model_input_buffer))) {
                    printf("Failed to capture input image\n");
                } else {
                    bool ok = run_inference_from_buffer(model_input_buffer, sizeof(model_input_buffer));
                    if (!ok) {
                        printf("Inference failed\n");
                    } else {
                        int pred = get_last_prediction();
                        float score = get_last_score();
                        printf("Prediction: %d (score: %.4f)\n", pred, score);

                        char buf[32];
                        snprintf(buf, sizeof(buf), "Pred: %d (%.2f)", pred, score);
                        GUI_DrawString_EN(10, 40, buf, &Font20, 0, 0);
                    }
                }

                while (!gpio_get(PIN_BUTTON)) sleep_ms(10);
                sleep_ms(300);
            }
        }

        sleep_ms(20);
    }

    return 0;
}
