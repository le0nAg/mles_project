﻿#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"
#include <stdio.h>
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "sd_utils.h"

const uint8_t bit_map[8][8]={
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 1, 1, 1, 1, 1, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
};

int main(void)
{
	uint8_t counter = 0;
	System_Init();
	LCD_SCAN_DIR  lcd_scan_dir = SCAN_DIR_DFT;
	LCD_Init(lcd_scan_dir,1000);
	TP_Init(lcd_scan_dir);
	LCD_SCAN_DIR bmp_scan_dir = D2U_R2L;
	TP_GetAdFac();
	TP_Dialog();

	uint16_t cnt=0;
	bool res = sd_write_async(bit_map, 8, 8, "file.txt");
	while (1) { 
		sleep_ms(1000); 
		printf("res: %d\n", res);
	}

	// while(1){
		
	// 	for(cnt=1000;cnt>2;cnt--)
	// 	{
	// 		LCD_SetBackLight(1000);
	// 		TP_DrawBoard();
	// 	}
	// }
	return 0;
}