#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "processing.h"

#define PI 3.14159265358979323846

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

static void explode_vertex_inplace(uint8_t* img, int w, int h, int16_t magnitude){
    // Work backwards to avoid overwriting pixels we haven't processed yet
    for(int y = h-1; y >= 0; y--){
        for(int x = w-1; x >= 0; x--){
            if(img[y * w + x] == 1){
                for(int dy = -1; dy <= 1; dy++){
                    for(int dx = -1; dx <= 1; dx++){
                        if(dy == 0 && dx == 0) continue;
                        for(int step = 1; step <= magnitude; step++){
                            int newY = y + dy * step;
                            int newX = x + dx * step;
                            if(newY >= 0 && newY < h && newX >= 0 && newX < w){
                                img[newY * w + newX] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

void connected_components(const uint8_t* img, int w, int h, int* n_components){
    // Use smaller stack allocation instead of malloc for small images
    uint8_t* img_exploded = (uint8_t*)malloc(w * h);
    if (!img_exploded) {
        *n_components = 0;
        return;
    }
    
    memcpy(img_exploded, img, w * h);
    explode_vertex_inplace(img_exploded, w, h, 1);
    
    // Use bit-packed visited array to save memory (8x reduction)
    size_t visited_bytes = (w * h + 7) / 8;
    uint8_t* visited = (uint8_t*)calloc(visited_bytes, 1);
    if (!visited) {
        free(img_exploded);
        *n_components = 0;
        return;
    }
    
    *n_components = 0;
    
    // Define stack size based on image size (reasonable upper bound)
    int max_stack_size = (w * h) / 4; // Conservative estimate
    if (max_stack_size < 256) max_stack_size = 256;
    if (max_stack_size > 4096) max_stack_size = 4096;
    
    int* stack_x = (int*)malloc(max_stack_size * sizeof(int));
    int* stack_y = (int*)malloc(max_stack_size * sizeof(int));
    
    if (!stack_x || !stack_y) {
        free(img_exploded);
        free(visited);
        if (stack_x) free(stack_x);
        if (stack_y) free(stack_y);
        *n_components = 0;
        return;
    }
    
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            int idx = y * w + x;
            int byte_idx = idx / 8;
            int bit_idx = idx % 8;
            
            if(img_exploded[idx] == 1 && !(visited[byte_idx] & (1 << bit_idx))){
                int stack_top = 0;
                
                stack_x[stack_top] = x;
                stack_y[stack_top] = y;
                stack_top++;
                
                while(stack_top > 0 && stack_top < max_stack_size){
                    stack_top--;
                    int curr_x = stack_x[stack_top];
                    int curr_y = stack_y[stack_top];
                    int curr_idx = curr_y * w + curr_x;
                    int curr_byte_idx = curr_idx / 8;
                    int curr_bit_idx = curr_idx % 8;
                    
                    if(curr_x < 0 || curr_x >= w || curr_y < 0 || curr_y >= h || 
                       (visited[curr_byte_idx] & (1 << curr_bit_idx)) || img_exploded[curr_idx] == 0){
                        continue;
                    }
                    
                    visited[curr_byte_idx] |= (1 << curr_bit_idx);
                    
                    // Add neighbors to stack if there's space
                    if (stack_top + 4 < max_stack_size) {
                        stack_x[stack_top] = curr_x + 1; stack_y[stack_top] = curr_y; stack_top++;
                        stack_x[stack_top] = curr_x - 1; stack_y[stack_top] = curr_y; stack_top++;
                        stack_x[stack_top] = curr_x; stack_y[stack_top] = curr_y + 1; stack_top++;
                        stack_x[stack_top] = curr_x; stack_y[stack_top] = curr_y - 1; stack_top++;
                    }
                }
                
                (*n_components)++;
            }
        }
    }
    
    free(stack_x);
    free(stack_y);
    free(img_exploded);
    free(visited);
}

void density(const uint8_t* img, int w, int h, float* density){
    int count_ones = 0;
    for(int i = 0; i < w * h; i++){
        if(img[i] == 1){
            count_ones++;
        }
    }
    *density = (float)count_ones / (w * h);
}

static void morphological_explode(const uint8_t* img, int w, int h, uint8_t* out){
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            uint8_t all_ones = 1;
            
            for (int dy = -1; dy <= 1 && all_ones; dy++) {
                for (int dx = -1; dx <= 1 && all_ones; dx++) {
                    int nidx = (y + dy) * w + (x + dx);
                    if (img[nidx] == 0) {
                        all_ones = 0;
                    }
                }
            }
            
            out[idx] = all_ones;
        }
    }
}

static void morphological_implode(const uint8_t* img, int w, int h, uint8_t* out){
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            uint8_t any_one = 0;
            
            for (int dy = -1; dy <= 1 && !any_one; dy++) {
                for (int dx = -1; dx <= 1 && !any_one; dx++) {
                    int nidx = (y + dy) * w + (x + dx);
                    if (img[nidx] == 1) {
                        any_one = 1;
                    }
                }
            }
            
            out[idx] = any_one;
        }
    }
}

void edge_detection(const uint8_t* img, int w, int h, uint8_t* out)
{
    if (!img || !out || w <= 0 || h <= 0) {
        printf("ERROR: Invalid parameters for edge_detection\r\n");
        return;
    }

    printf("edge_detection: Processing in-place to save memory\r\n");
    
    // Use output buffer as working space to avoid extra allocation
    morphological_implode(img, w, h, out);
    
    // Process edges in-place
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            // Compute erosion on the fly
            uint8_t all_ones = 1;
            for (int dy = -1; dy <= 1 && all_ones; dy++) {
                for (int dx = -1; dx <= 1 && all_ones; dx++) {
                    int nidx = (y + dy) * w + (x + dx);
                    if (img[nidx] == 0) {
                        all_ones = 0;
                    }
                }
            }
            
            // Edge = Dilation - Erosion
            out[idx] = out[idx] - all_ones;
        }
    }
    
    printf("edge_detection: Complete\r\n");
}

static int area(const uint8_t* img, int w, int h){
    int area = 0;
    for(int i = 0; i < w * h; i++){
        if(img[i] == 1){
            area++;
        }
    }
    return area;
}

static int perimeter(const uint8_t* img, int w, int h){
    int perimeter = 0;
    
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            int idx = y * w + x;
            if(img[idx] == 1){
                // Check 4-connected neighbors
                if(x == 0 || img[y * w + (x - 1)] == 0) perimeter++; // Left
                if(x == w - 1 || img[y * w + (x + 1)] == 0) perimeter++; // Right
                if(y == 0 || img[(y - 1) * w + x] == 0) perimeter++; // Top
                if(y == h - 1 || img[(y + 1) * w + x] == 0) perimeter++; // Bottom
            }
        }
    }
    
    return perimeter;
}

void compactness(const uint8_t* img, int w, int h, float* isoperimetric, float* a_to_p_ratio, float* circularity_ratio){
    int a = area(img, w, h);
    int p = perimeter(img, w, h);
    
    if(p == 0){
        *isoperimetric = 0.0f;
        *a_to_p_ratio = 0.0f;
        *circularity_ratio = 0.0f;
        return;
    }

    *isoperimetric = (float)(4 * PI * a) / (p * p);
    *a_to_p_ratio = (float)a / (float)p;
    *circularity_ratio = (float)(p * p) / (4 * PI * a);
}

bool extract_features_json(const uint8_t* img, int w, int h, char* out_json, size_t json_buffer_size)
{
    if (!img || !out_json || json_buffer_size < 512) {
        printf("ERROR: Invalid parameters for extract_features_json\r\n");
        return false;
    }

    printf("extract_features_json: Starting optimized feature extraction (%dx%d)\r\n", w, h);

    // Use single buffer for edge detection (reuse output buffer)
    uint8_t* edges = (uint8_t*)malloc(w * h);
    if (!edges) {
        printf("ERROR: Failed to allocate memory for edge detection\r\n");
        return false;
    }

    // Extract features
    printf("extract_features_json: Computing connected components...\r\n");
    int n_components = 0;
    connected_components(img, w, h, &n_components);

    printf("extract_features_json: Computing density...\r\n");
    float pixel_density = 0.0f;
    density(img, w, h, &pixel_density);

    printf("extract_features_json: Running edge detection...\r\n");
    edge_detection(img, w, h, edges);
    
    int edge_count = 0;
    for (int i = 0; i < w * h; i++) {
        if (edges[i] == 1) edge_count++;
    }

    printf("extract_features_json: Computing compactness...\r\n");
    float isoperimetric = 0.0f;
    float a_to_p_ratio = 0.0f;
    float circularity_ratio = 0.0f;
    compactness(img, w, h, &isoperimetric, &a_to_p_ratio, &circularity_ratio);

    free(edges);

    printf("extract_features_json: Building JSON string...\r\n");
    
    // Build JSON string
    int written = snprintf(out_json, json_buffer_size,
        "{\n"
        "  \"features\": {\n"
        "    \"image_dimensions\": {\n"
        "      \"width\": %d,\n"
        "      \"height\": %d\n"
        "    },\n"
        "    \"edges\": %d,\n"
        "    \"pixel_density\": %.6f,\n"
        "    \"connected_components\": %d,\n"
        "    \"compactness\": {\n"
        "      \"isoperimetric_quotient\": %.6f,\n"
        "      \"area_to_perimeter_ratio\": %.6f,\n"
        "      \"circularity_ratio\": %.6f\n"
        "    }\n"
        "  }\n"
        "}",
        w, h,
        edge_count,
        pixel_density,
        n_components,
        isoperimetric,
        a_to_p_ratio,
        circularity_ratio
    );

    if (written < 0 || (size_t)written >= json_buffer_size) {
        printf("ERROR: JSON buffer too small (needed %d bytes)\r\n", written);
        return false;
    }

    printf("extract_features_json: Complete (%d bytes)\r\n", written);
    return true;
}

void print_free_memory(void) {
    extern char __StackLimit, __bss_end__;
    int free_mem = &__StackLimit - &__bss_end__;
    printf("Free memory: ~%d bytes\r\n", free_mem);
}