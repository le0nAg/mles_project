/**
 * @file digit_inference.cc
 * @brief TFLite Micro inference implementation for Pico
 */

#include "digit_inference.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Tensor arena size - adjust based on model requirements
// MCUNet tiny needs ~20KB, add some buffer
constexpr int kTensorArenaSize = 24 * 1024;

// Static allocations to avoid heap fragmentation
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static bool is_initialized = false;

// Op resolver - add only ops used by the model
static tflite::MicroMutableOpResolver<10> resolver;

extern "C" {

int digit_inference_init(void) {
    if (is_initialized) {
        return DIGIT_OK;
    }

    // Load model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch: %d vs %d",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return DIGIT_ERR_INIT;
    }

    // Add operations used by MCUNet tiny
    // Adjust this list based on your model's actual ops
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddFullyConnected();
    resolver.AddMean();           // For GlobalAveragePooling
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return DIGIT_ERR_INIT;
    }

    // Get input/output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Verify input shape
    if (input_tensor->dims->size != 4 ||
        input_tensor->dims->data[1] != DIGIT_IMG_HEIGHT ||
        input_tensor->dims->data[2] != DIGIT_IMG_WIDTH ||
        input_tensor->dims->data[3] != 1) {
        MicroPrintf("Unexpected input shape");
        return DIGIT_ERR_INIT;
    }

    // Verify output shape
    if (output_tensor->dims->size != 2 ||
        output_tensor->dims->data[1] != DIGIT_NUM_CLASSES) {
        MicroPrintf("Unexpected output shape");
        return DIGIT_ERR_INIT;
    }

    is_initialized = true;
    
    MicroPrintf("Digit inference initialized. Arena used: %d bytes",
                interpreter->arena_used_bytes());
    
    return DIGIT_OK;
}

int digit_inference_predict(const uint8_t* image_28x28, uint8_t* confidence) {
    if (!is_initialized) {
        return DIGIT_ERR_INIT;
    }
    
    if (image_28x28 == nullptr) {
        return DIGIT_ERR_NULL_INPUT;
    }

    // Copy input data
    // Model expects uint8 input (0-255)
    if (input_tensor->type == kTfLiteUInt8) {
        uint8_t* input_data = input_tensor->data.uint8;
        for (int i = 0; i < DIGIT_IMG_SIZE; i++) {
            input_data[i] = image_28x28[i];
        }
    } else if (input_tensor->type == kTfLiteInt8) {
        // Convert uint8 [0,255] to int8 [-128,127]
        int8_t* input_data = input_tensor->data.int8;
        for (int i = 0; i < DIGIT_IMG_SIZE; i++) {
            input_data[i] = (int8_t)(image_28x28[i] - 128);
        }
    } else {
        MicroPrintf("Unsupported input type: %d", input_tensor->type);
        return DIGIT_ERR_INIT;
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return DIGIT_ERR_INVOKE;
    }

    // Find argmax of output
    int predicted_digit = 0;
    int8_t max_score = -128;
    
    if (output_tensor->type == kTfLiteInt8) {
        int8_t* output_data = output_tensor->data.int8;
        for (int i = 0; i < DIGIT_NUM_CLASSES; i++) {
            if (output_data[i] > max_score) {
                max_score = output_data[i];
                predicted_digit = i;
            }
        }
        
        // Convert score to confidence (0-255)
        if (confidence != nullptr) {
            // Map [-128, 127] to [0, 255]
            *confidence = (uint8_t)(max_score + 128);
        }
    } else if (output_tensor->type == kTfLiteUInt8) {
        uint8_t* output_data = output_tensor->data.uint8;
        uint8_t max_val = 0;
        for (int i = 0; i < DIGIT_NUM_CLASSES; i++) {
            if (output_data[i] > max_val) {
                max_val = output_data[i];
                predicted_digit = i;
            }
        }
        if (confidence != nullptr) {
            *confidence = max_val;
        }
    }

    return predicted_digit;
}

int digit_inference_get_scores(const uint8_t* image_28x28, int8_t* scores) {
    if (!is_initialized) {
        return DIGIT_ERR_INIT;
    }
    
    if (image_28x28 == nullptr || scores == nullptr) {
        return DIGIT_ERR_NULL_INPUT;
    }

    // Copy input
    if (input_tensor->type == kTfLiteUInt8) {
        uint8_t* input_data = input_tensor->data.uint8;
        for (int i = 0; i < DIGIT_IMG_SIZE; i++) {
            input_data[i] = image_28x28[i];
        }
    } else if (input_tensor->type == kTfLiteInt8) {
        int8_t* input_data = input_tensor->data.int8;
        for (int i = 0; i < DIGIT_IMG_SIZE; i++) {
            input_data[i] = (int8_t)(image_28x28[i] - 128);
        }
    }

    // Invoke
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        return DIGIT_ERR_INVOKE;
    }

    if (output_tensor->type == kTfLiteInt8) {
        int8_t* output_data = output_tensor->data.int8;
        for (int i = 0; i < DIGIT_NUM_CLASSES; i++) {
            scores[i] = output_data[i];
        }
    } else if (output_tensor->type == kTfLiteUInt8) {
        uint8_t* output_data = output_tensor->data.uint8;
        for (int i = 0; i < DIGIT_NUM_CLASSES; i++) {
            scores[i] = (int8_t)(output_data[i] - 128);
        }
    }

    return DIGIT_OK;
}

bool digit_inference_is_ready(void) {
    return is_initialized;
}

void digit_inference_free(void) {
    // Static allocation - nothing to free
    // Just reset state
    is_initialized = false;
    interpreter = nullptr;
    input_tensor = nullptr;
    output_tensor = nullptr;
}

const char* digit_inference_get_info(void) {
    static char info[128];
    if (is_initialized && interpreter != nullptr) {
        snprintf(info, sizeof(info),
                 "MCUNet-tiny INT8, arena: %d/%d bytes",
                 (int)interpreter->arena_used_bytes(), kTensorArenaSize);
    } else {
        snprintf(info, sizeof(info), "Not initialized");
    }
    return info;
}

}