#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"
#include "digit_collector.h"
#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/watchdog.h"

// Set to 1 to disable SD card for testing (helps isolate screen issue)
#define DISABLE_SD_FOR_TESTING  0

// Set to 1 to enable watchdog (will reset if system hangs)
#define ENABLE_WATCHDOG         1

// Watchdog timeout in milliseconds
#define WATCHDOG_TIMEOUT_MS     8000

int main(void) {
    // Initialize stdio for debug output
    stdio_init_all();
    
    // Wait for USB serial to connect (optional, helps with debugging)
    sleep_ms(2000);
    
    printf("\n\n");
    printf("========================================\n");
    printf("=== Digit Data Collector Starting... ===\n");
    printf("========================================\n");
    
    // Check if we're recovering from a watchdog reset
    if (watchdog_caused_reboot()) {
        printf("WARNING: System was reset by watchdog!\n");
        printf("This indicates a hang occurred in the previous run.\n");
    }
    
    printf("Initializing system...\n");
    System_Init();
    printf("System_Init complete\n");
    
    printf("Initializing LCD...\n");
    LCD_SCAN_DIR lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir, 1000);
    printf("LCD_Init complete\n");
    
    printf("Initializing touch panel...\n");
    TP_Init(lcd_scan_dir);
    TP_GetAdFac();
    printf("Touch panel init complete\n");
    
#if ENABLE_WATCHDOG
    // Enable watchdog - will reset if not fed within timeout
    printf("Enabling watchdog with %d ms timeout...\n", WATCHDOG_TIMEOUT_MS);
    watchdog_enable(WATCHDOG_TIMEOUT_MS, true);
#endif
    
    printf("Initializing collector...\n");
    Collector_Init();
    printf("Collector_Init complete\n");
    
#if ENABLE_WATCHDOG
    watchdog_update();  // Feed watchdog
#endif
    
    printf("Drawing screen...\n");
    
#if DISABLE_SD_FOR_TESTING
    printf("NOTE: SD card disabled for testing\n");
    Collector_DrawScreen_NoSD();
#else
    Collector_DrawScreen();
#endif
    
    printf("Screen draw complete\n");
    
#if ENABLE_WATCHDOG
    watchdog_update();  // Feed watchdog
#endif
    
    printf("\n");
    printf("=== Entering main loop ===\n");
    printf("Draw digits in the box and press SAVE\n");
    printf("Use < > buttons to change digit (0-9)\n");
    printf("\n");
    
    uint32_t loop_count = 0;
    uint32_t last_status_time = to_ms_since_boot(get_absolute_time());
    
    while (1) {
        // Run the collector
        Collector_Run();
        
        // Small delay to prevent overwhelming the system
        sleep_ms(10);
        
#if ENABLE_WATCHDOG
        // Feed the watchdog to prevent reset
        watchdog_update();
#endif
        
        loop_count++;
        
        // Status output and keep-alive every 5 seconds
        uint32_t now = to_ms_since_boot(get_absolute_time());
        if (now - last_status_time >= 5000) {
            printf("Loop running: %lu iterations, SD: %s\n", 
                   loop_count, 
                   Collector_IsSDReady() ? "Ready" : "Not Ready");
            
            // Call keep-alive to prevent display timeout
            Collector_KeepAlive();
            
            last_status_time = now;
        }
    }
    
    return 0;
}