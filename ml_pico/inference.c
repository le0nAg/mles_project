#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "fc.cc"

#ifndef MODEL_DATA
extern const unsigned char model_final_pico_tflite[];
extern const unsigned int model_final_pico_tflite_len;
#define MODEL_DATA model_final_pico_tflite
#define MODEL_SIZE model_final_pico_tflite_len
#endif

#ifndef MODEL_DATA
extern const unsigned char model_tflite[];
extern const unsigned int model_tflite_len;
#define MODEL_DATA model_tflite
#define MODEL_SIZE model_tflite_len
#endif

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (70 * 1024)
#endif

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static const tflite::Model *g_model = NULL;
static tflite::MicroInterpreter *g_interpreter = NULL;
static TfLiteTensor *g_input = NULL;
static TfLiteTensor *g_output = NULL;
static int g_num_classes = 0;

static int g_last_prediction = -1;
static float g_last_score = 0.0f;

static tflite::AllOpsResolver resolver;

bool init_inference()
{
    g_model = tflite::GetModel(MODEL_DATA);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version %d not equal to supported version %d\n",
               g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr);
    g_interpreter = &static_interpreter;

    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return false;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    if (g_output->dims->size >= 1) {
        g_num_classes = g_output->dims->data[g_output->dims->size - 1];
    } else {
        g_num_classes = 1;
    }

    printf("TFLite Micro initialized. Input bytes: %d, output classes: %d\n",
           g_input->bytes, g_num_classes);

    return true;
}

static void zero_input_tensor()
{
    memset(g_input->data.uint8, 0, g_input->bytes);
}

bool run_inference_from_buffer(const void *input_buf, size_t input_size)
{
    if (!g_interpreter || !g_input) return false;

    if (input_size != (size_t)g_input->bytes) {
        size_t to_copy = input_size < (size_t)g_input->bytes ? input_size : (size_t)g_input->bytes;
        memcpy(g_input->data.uint8, input_buf, to_copy);
        if (to_copy < (size_t)g_input->bytes) {
            memset(((uint8_t*)g_input->data.uint8) + to_copy, 0, g_input->bytes - to_copy);
        }
    } else {
        memcpy(g_input->data.uint8, input_buf, input_size);
    }

    TfLiteStatus invoke_status = g_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke failed: %d\n", invoke_status);
        return false;
    }

    int predicted = 0;
    float best_score = -1e9;

    if (g_output->type == kTfLiteFloat32) {
        float *out = g_output->data.f;
        for (int i = 0; i < g_num_classes; ++i) {
            float v = out[i];
            if (v > best_score) {
                best_score = v;
                predicted = i;
            }
        }
    } else if (g_output->type == kTfLiteUInt8) {
        const float scale = g_output->params.scale;
        const int zero_point = g_output->params.zero_point;
        uint8_t *out = g_output->data.uint8;
        for (int i = 0; i < g_num_classes; ++i) {
            float v = (out[i] - zero_point) * scale;
            if (v > best_score) {
                best_score = v;
                predicted = i;
            }
        }
    } else if (g_output->type == kTfLiteInt8) {
        const float scale = g_output->params.scale;
        const int zero_point = g_output->params.zero_point;
        int8_t *out = g_output->data.int8;
        for (int i = 0; i < g_num_classes; ++i) {
            float v = (out[i] - zero_point) * scale;
            if (v > best_score) {
                best_score = v;
                predicted = i;
            }
        }
    } else {
        uint8_t *out = g_output->data.uint8;
        for (int i = 0; i < g_num_classes; ++i) {
            float v = out[i];
            if (v > best_score) {
                best_score = v;
                predicted = i;
            }
        }
    }

    g_last_prediction = predicted;
    g_last_score = best_score;

    return true;
}

int get_last_prediction()
{
    return g_last_prediction;
}

float get_last_score()
{
    return g_last_score;
}
