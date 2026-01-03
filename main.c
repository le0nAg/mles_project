#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"
#include "pin_auth_ui.h"
#include <stdio.h>
#include "pico/stdlib.h"

int main(void) {
    System_Init();
    
    LCD_SCAN_DIR lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir, 1000);
    TP_Init(lcd_scan_dir);
    TP_GetAdFac();
    
    printf("\n=== PIN Authentication ===\n");
    
    PinAuth_Init();
    PinAuth_DrawScreen();
    
    while (1) {
        PinAuth_Run();
        sleep_ms(10);
    }
    
    return 0;
}