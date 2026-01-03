#ifndef DIGIT_COLLECTOR_H
#define DIGIT_COLLECTOR_H

#include <stdbool.h>

// Initialize the collector state
void Collector_Init(void);

// Draw the screen and initialize SD card
void Collector_DrawScreen(void);

// Draw the screen WITHOUT SD card init (for testing)
void Collector_DrawScreen_NoSD(void);

// Main run loop - call this repeatedly
void Collector_Run(void);

// Call periodically to prevent display timeout (optional)
void Collector_KeepAlive(void);

// Check if SD card is ready
bool Collector_IsSDReady(void);

#endif