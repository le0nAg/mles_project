#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "processing.h"

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


/**
 * @docs: Explode vertex pixels (1s) in all 8 directions by a given magnitude.
 */
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