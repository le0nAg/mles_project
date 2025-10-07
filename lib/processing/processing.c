#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

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


static void explode_vertex(const uint8_t* img, int w, int h, uint8_t* out, int16_t magnitude){
    for(int i = 0; i < w * h; i++){
        out[i] = img[i];
    }

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            if(img[y * w + x] == 1){
                //  Explode in all 8 directions
                //  TODO: consider a loop unrolling for performance
                for(int dy = -1; dy <= 1; dy++){
                    for(int dx = -1; dx <= 1; dx++){
                        if(dy == 0 && dx == 0) continue; // Skip the center pixel
                        for(int step = 1; step <= magnitude; step++){
                            int newY = y + dy * step;
                            int newX = x + dx * step;
                            if(newY >= 0 && newY < h && newX >= 0 && newX < w){
                                out[newY * w + newX] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

void connected_components(const uint8_t* img, int w, int h, int* n_components){
    uint8_t* img_exploded = (uint8_t*)malloc(w * h);
    explode_vertex(img, w, h, img_exploded, 1);
    
    bool* visited = (bool*)calloc(w * h, sizeof(bool));
    *n_components = 0;
    
    // Flood fill for each unvisited pixel
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            int idx = y * w + x;
            if(img_exploded[idx] == 1 && !visited[idx]){
                int* stack_x = (int*)malloc(w * h * sizeof(int));
                int* stack_y = (int*)malloc(w * h * sizeof(int));
                int stack_top = 0;
                
                stack_x[stack_top] = x;
                stack_y[stack_top] = y;
                stack_top++;
                
                while(stack_top > 0){
                    stack_top--;
                    int curr_x = stack_x[stack_top];
                    int curr_y = stack_y[stack_top];
                    int curr_idx = curr_y * w + curr_x;
                    
                    if(curr_x < 0 || curr_x >= w || curr_y < 0 || curr_y >= h || 
                       visited[curr_idx] || img_exploded[curr_idx] == 0){
                        continue;
                    }
                    
                    visited[curr_idx] = true;
                    
                    stack_x[stack_top] = curr_x + 1; stack_y[stack_top] = curr_y; stack_top++;
                    stack_x[stack_top] = curr_x - 1; stack_y[stack_top] = curr_y; stack_top++;
                    stack_x[stack_top] = curr_x; stack_y[stack_top] = curr_y + 1; stack_top++;
                    stack_x[stack_top] = curr_x; stack_y[stack_top] = curr_y - 1; stack_top++;
                }
                
                free(stack_x);
                free(stack_y);
                (*n_components)++;
            }
        }
    }
    
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
    memset(out, 0, w * h);
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            uint8_t all_ones = 1;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    int nidx = ny * w + nx;
                    
                    if (img[nidx] == 0) {
                        all_ones = 0;
                        break;
                    }
                }
                if (!all_ones) break;
            }
            
            out[idx] = all_ones;
        }
    }
}

static void morphological_implode(const uint8_t* img, int w, int h, uint8_t* out){
    memset(out, 0, w * h);
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            
            uint8_t any_one = 0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    int nidx = ny * w + nx;
                    
                    if (img[nidx] == 1) {
                        any_one = 1;
                        break;
                    }
                }
                if (any_one) break;
            }
            
            out[idx] = any_one;
        }
    }
}

void edge_detection(const uint8_t* img, int w, int h, uint8_t* out){
    uint8_t* exploded = (uint8_t*)malloc(w * h);
    uint8_t* imploded = (uint8_t*)malloc(w * h);
    
    if (!exploded || !imploded) {
        memcpy(out, img, w * h);
        free(exploded);
        free(imploded);
        return;
    }
    
    // Step 1: Erode the image (shrinks objects)
    morphological_explode(img, w, h, exploded);
    
    // Step 2: Dilate the image (expands objects)
    morphological_implode(img, w, h, imploded);
    
    // Step 3: Edge = Dilation - Erosion
    // This gives us both inner and outer boundaries
    for (int i = 0; i < w * h; i++) {
        out[i] = imploded[i] - exploded[i];
    }
    
    free(exploded);
    free(imploded);
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

    *isoperimetric = (float)(4 * PI * a) / (p * p); // 4πA/P²
    *a_to_p_ratio = (float)a / (float)p; // A/P
    *circularity_ratio = (float)(p * p) / (4 * PI * a); // P²/4πA
}

bool extract_features_json(const uint8_t* img, int w, int h, char* out_json, size_t json_buffer_size){
    if (!img || !out_json || json_buffer_size < 512) {
        printf("ERROR: Invalid parameters for extract_features_json\r\n");
        return false;
    }

    uint8_t* edges = (uint8_t*)malloc(w * h);
    if (!edges) {
        printf("ERROR: Failed to allocate memory for edge detection\r\n");
        return false;
    }

    // Extract features
    int n_components = 0;
    connected_components(img, w, h, &n_components);

    float pixel_density = 0.0f;
    density(img, w, h, &pixel_density);

    edge_detection(img, w, h, edges);
    int edge_count = 0;
    for (int i = 0; i < w * h; i++) {
        if (edges[i] == 1) edge_count++;
    }

    float isoperimetric = 0.0f;
    float a_to_p_ratio = 0.0f;
    float circularity_ratio = 0.0f;
    compactness(img, w, h, &isoperimetric, &a_to_p_ratio, &circularity_ratio);

    free(edges);

    // Build JSON
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
        w, h, edge_count, pixel_density, n_components,
        isoperimetric, a_to_p_ratio, circularity_ratio
    );

    if (written < 0 || (size_t)written >= json_buffer_size) {
        printf("ERROR: JSON buffer too small\r\n");
        return false;
    }

    return true;
}