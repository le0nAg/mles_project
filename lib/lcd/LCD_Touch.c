#include "LCD_Touch.h"

#include "sd_utils.h"

#include <stdlib.h>
#include <string.h>  
#include <stdio.h>

#include "processing.h"

extern LCD_DIS sLCD_DIS;
extern uint8_t id;
static TP_DEV sTP_DEV;
static TP_DRAW sTP_Draw;


// ---- Capture area (the black box) ------------------------------------------
// #define BOX_X0 100
// #define BOX_Y0 50
// #define BOX_X1 380   // right  edge (exclusive in our math)
// #define BOX_Y1 290   // bottom edge (exclusive in our math)

// #define BOX_W (BOX_X1 - BOX_X0)   // 280 -> 280/8 = 35 per row after bit packing
// #define BOX_H (BOX_Y1 - BOX_Y0)   // 240 -> 240/8 = 30 per column after bit packing

#define BOX_X0 252
#define BOX_Y0 50
#define BOX_X1 380   // right edge fixed
#define BOX_Y1 162   // bottom edge (exclusive in our math)

#define BOX_W (BOX_X1 - BOX_X0)   // 128 -> 128/8 = 16 per row after bit packing
#define BOX_H (BOX_Y1 - BOX_Y0)   // 112 -> 112/8 = 14 per column after bit packing


// Live "shadow" of what has been drawn (0 = empty, 1 = drawn)
static uint8_t sDrawShadow[BOX_H][BOX_W];

// Snapshot taken when SAVE is pressed (this is what you'll write to SD later)
static uint8_t sSavedBitmap[BOX_H][BOX_W];

// Optional: quick accessor if you want to use these elsewhere
uint16_t TP_SavedWidth(void)  { return BOX_W; }
uint16_t TP_SavedHeight(void) { return BOX_H; }
const uint8_t* TP_SavedData(void) { return &sSavedBitmap[0][0]; }

// Helper to record a pixel into the shadow buffer
static inline void Capture_SetPixel(uint16_t x, uint16_t y)
{
    if (x >= BOX_X0 && x < BOX_X1 && y >= BOX_Y0 && y < BOX_Y1) {
        sDrawShadow[y - BOX_Y0][x - BOX_X0] = 1;
    }
}


static void TP_DumpBitmapToSerial(const uint8_t bmp[BOX_H][BOX_W])
{
    // Build and print one line at a time (faster than putchar per pixel)
    char line[BOX_W + 2];           // + '\n' + '\0'
    line[BOX_W]   = '\n';
    line[BOX_W+1] = '\0';

    for (uint16_t y = 0; y < BOX_H; ++y) {
        for (uint16_t x = 0; x < BOX_W; ++x) {
            line[x] = bmp[y][x] ? '1' : '0';
        }
        printf("%s", line);         // prints 280 chars of 0/1, then newline
    }
}

static uint16_t TP_Read_ADC(uint8_t CMD)
{
    uint16_t Data = 0;

    //A cycle of at least 400ns.
    DEV_Digital_Write(TP_CS_PIN, 0);

    SPI4W_Write_Byte(CMD);
    Driver_Delay_us(200);

    //	dont write 0xff, it will block xpt2046
    //Data = SPI4W_Read_Byte(0Xff);
    Data = SPI4W_Read_Byte(0X00);
    Data <<= 8; //7bit
    Data |= SPI4W_Read_Byte(0X00);
    //Data = SPI4W_Read_Byte(0Xff);
    Data >>= 3; //5bit
    DEV_Digital_Write(TP_CS_PIN, 1);
    return Data;
}

#define READ_TIMES 5 //Number of readings
#define LOST_NUM 1   //Discard value
static uint16_t 
TP_Read_ADC_Average(uint8_t Channel_Cmd)
{
    uint8_t i, j;
    uint16_t Read_Buff[READ_TIMES];
    uint16_t Read_Sum = 0, Read_Temp = 0;
    //LCD SPI speed = 3 MHz
    spi_set_baudrate(SPI_PORT, 3000000);
    //Read and save multiple samples
    for (i = 0; i < READ_TIMES; i++)
    {
        Read_Buff[i] = TP_Read_ADC(Channel_Cmd);
        Driver_Delay_us(200);
    }
    //LCD SPI speed = 18 MHz
    spi_set_baudrate(SPI_PORT, 18000000);
    //Sort from small to large
    for (i = 0; i < READ_TIMES - 1; i++)
    {
        for (j = i + 1; j < READ_TIMES; j++)
        {
            if (Read_Buff[i] > Read_Buff[j])
            {
                Read_Temp = Read_Buff[i];
                Read_Buff[i] = Read_Buff[j];
                Read_Buff[j] = Read_Temp;
            }
        }
    }

    //Exclude the largest and the smallest
    for (i = LOST_NUM; i < READ_TIMES - LOST_NUM; i++)
        Read_Sum += Read_Buff[i];

    //Averaging
    Read_Temp = Read_Sum / (READ_TIMES - 2 * LOST_NUM);

    return Read_Temp;
}

static void TP_Read_ADC_XY(uint16_t *pXCh_Adc, uint16_t *pYCh_Adc)
{
    *pXCh_Adc = TP_Read_ADC_Average(0xD0);
    *pYCh_Adc = TP_Read_ADC_Average(0x90);
}

#define ERR_RANGE 50 //tolerance scope
static bool TP_Read_TwiceADC(uint16_t *pXCh_Adc, uint16_t *pYCh_Adc)
{
    uint16_t XCh_Adc1, YCh_Adc1, XCh_Adc2, YCh_Adc2;

    //Read the ADC values Read the ADC values twice
    TP_Read_ADC_XY(&XCh_Adc1, &YCh_Adc1);
    Driver_Delay_us(10);
    TP_Read_ADC_XY(&XCh_Adc2, &YCh_Adc2);
    Driver_Delay_us(10);

    //The ADC error used twice is greater than ERR_RANGE to take the average
    if (((XCh_Adc2 <= XCh_Adc1 && XCh_Adc1 < XCh_Adc2 + ERR_RANGE) ||
         (XCh_Adc1 <= XCh_Adc2 && XCh_Adc2 < XCh_Adc1 + ERR_RANGE)) &&
        ((YCh_Adc2 <= YCh_Adc1 && YCh_Adc1 < YCh_Adc2 + ERR_RANGE) ||
         (YCh_Adc1 <= YCh_Adc2 && YCh_Adc2 < YCh_Adc1 + ERR_RANGE)))
    {
        *pXCh_Adc = (XCh_Adc1 + XCh_Adc2) / 2;
        *pYCh_Adc = (YCh_Adc1 + YCh_Adc2) / 2;
        return true;
    }

    //The ADC error used twice is less than ERR_RANGE returns failed
    return false;
}

static uint8_t TP_Scan(uint8_t chCoordType)
{
    //In X, Y coordinate measurement, IRQ is disabled and output is low
    if (!DEV_Digital_Read(TP_IRQ_PIN))
    { //Press the button to press
        //Read the physical coordinates
        if (chCoordType)
        {
            TP_Read_TwiceADC(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint);
            //Read the screen coordinates
        }
        else if (TP_Read_TwiceADC(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint))
        {

            if (LCD_2_8 == id)
            {
                sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                  sTP_DEV.fXfac * sTP_DEV.Xpoint -
                                  sTP_DEV.iXoff;
                sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                  sTP_DEV.fYfac * sTP_DEV.Ypoint -
                                  sTP_DEV.iYoff;
            }
            else
            {
                //DEBUG("(Xad,Yad) = %d,%d\r\n",sTP_DEV.Xpoint,sTP_DEV.Ypoint);
                if (sTP_DEV.TP_Scan_Dir == R2L_D2U)
                { //Converts the result to screen coordinates
                    sTP_Draw.Xpoint = sTP_DEV.fXfac * sTP_DEV.Xpoint +
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sTP_DEV.fYfac * sTP_DEV.Ypoint +
                                      sTP_DEV.iYoff;
                }
                else if (sTP_DEV.TP_Scan_Dir == L2R_U2D)
                {
                    sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                      sTP_DEV.fXfac * sTP_DEV.Xpoint -
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                      sTP_DEV.fYfac * sTP_DEV.Ypoint -
                                      sTP_DEV.iYoff;
                }
                else if (sTP_DEV.TP_Scan_Dir == U2D_R2L)
                {
                    sTP_Draw.Xpoint = sTP_DEV.fXfac * sTP_DEV.Ypoint +
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sTP_DEV.fYfac * sTP_DEV.Xpoint +
                                      sTP_DEV.iYoff;
                }
                else
                {
                    sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                      sTP_DEV.fXfac * sTP_DEV.Ypoint -
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                      sTP_DEV.fYfac * sTP_DEV.Xpoint -
                                      sTP_DEV.iYoff;
                }
                // DEBUG("( x , y ) = %d,%d\r\n",sTP_Draw.Xpoint,sTP_Draw.Ypoint);
            }
        }
        if (0 == (sTP_DEV.chStatus & TP_PRESS_DOWN))
        { //Not being pressed
            sTP_DEV.chStatus = TP_PRESS_DOWN | TP_PRESSED;
            sTP_DEV.Xpoint0 = sTP_DEV.Xpoint;
            sTP_DEV.Ypoint0 = sTP_DEV.Ypoint;
        }
    }
    else
    {
        if (sTP_DEV.chStatus & TP_PRESS_DOWN)
        {                                  //0x80
            sTP_DEV.chStatus &= ~(1 << 7); //0x00
        }
        else
        {
            sTP_DEV.Xpoint0 = 0;
            sTP_DEV.Ypoint0 = 0;
            sTP_DEV.Xpoint = 0xffff;
            sTP_DEV.Ypoint = 0xffff;
        }
    }

    return (sTP_DEV.chStatus & TP_PRESS_DOWN);
}

void TP_GetAdFac(void)
{
    if (LCD_2_8 == id)
    {
        sTP_DEV.fXfac = 0.066626;
        sTP_DEV.fYfac = 0.089779;
        sTP_DEV.iXoff = -20;
        sTP_DEV.iYoff = -34;
    }
    else
    {
        if (sTP_DEV.TP_Scan_Dir == D2U_L2R)
        { //SCAN_DIR_DFT = D2U_L2R
            sTP_DEV.fXfac = -0.132443;
            sTP_DEV.fYfac = 0.089997;
            sTP_DEV.iXoff = 516;
            sTP_DEV.iYoff = -22;
        }
        else if (sTP_DEV.TP_Scan_Dir == L2R_U2D)
        {
            sTP_DEV.fXfac = 0.089697;
            sTP_DEV.fYfac = 0.134792;
            sTP_DEV.iXoff = -21;
            sTP_DEV.iYoff = -39;
        }
        else if (sTP_DEV.TP_Scan_Dir == R2L_D2U)
        {
            sTP_DEV.fXfac = 0.089915;
            sTP_DEV.fYfac = 0.133178;
            sTP_DEV.iXoff = -22;
            sTP_DEV.iYoff = -38;
        }
        else if (sTP_DEV.TP_Scan_Dir == U2D_R2L)
        {
            sTP_DEV.fXfac = -0.132906;
            sTP_DEV.fYfac = 0.087964;
            sTP_DEV.iXoff = 517;
            sTP_DEV.iYoff = -20;
        }
        else
        {
            LCD_Clear(LCD_BACKGROUND);
            GUI_DisString_EN(0, 60, "Does not support touch-screen \
							calibration in this direction",
                             &Font16, FONT_BACKGROUND, RED);
        }
    }
}


void TP_Dialog(void)
{
    LCD_Clear(LCD_BACKGROUND);

    GUI_DisString_EN(sLCD_DIS.LCD_Dis_Column - 60, 0,
                        "CLEAR", &Font16, BLACK, WHITE);
    GUI_DisString_EN(sLCD_DIS.LCD_Dis_Column - 120, 0,
                        "SAVE", &Font16, BLACK, WHITE);

    // Draw a box with black border
    GUI_DrawRectangle(BOX_X0 , BOX_Y0 ,
                      BOX_X1, BOX_Y1,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_2X2);

    // NEW: also clear the shadow buffer that mirrors what's drawn
    memset(sDrawShadow, 0, sizeof(sDrawShadow));
}

/** 
 * @leo 
 * Pipeline:
 * 1. Capture current drawing to saved bitmap
 * 2. Save original packed bitmap as <uuid>.bim
 * 3. Apply edge detection and save as <uuid>.morph.bim
 * 4. Extract features and save as <uuid>.json
 */
#define DEBUG_PRINT
void TP_Save(void)
{
    // Step 1: Snapshot current drawing into the "saved" buffer
    memcpy(sSavedBitmap, sDrawShadow, sizeof(sSavedBitmap));

    #ifdef DEBUG_PRINT
    int pixel_count = 0;
    for (int y = 0; y < BOX_H; y++) {
        for (int x = 0; x < BOX_W; x++) {
            if (sSavedBitmap[y][x]) pixel_count++;
        }
    }
    printf("TP_Save: Found %d drawn pixels\r\n", pixel_count);
    #endif

    // Generate UUID for this capture
    char uuid[37];
    generate_uuid(uuid);
    printf("TP_Save: Generated UUID: %s\r\n", uuid);

    // Save original packed bitmap
    char filename_original[64];
    snprintf(filename_original, sizeof(filename_original), "/%s.bim", uuid);

    bool res_original = sd_write_async_packed((uint8_t*)sSavedBitmap, BOX_W, BOX_H, filename_original);
    if (res_original) {
        printf("TP_Save: Queued original bitmap write to '%s'\r\n", filename_original);
    } else {
        printf("TP_Save: Failed to queue original bitmap write\r\n");
        goto user_feedback;
    }

    // Crop the image
    int out_w, out_h;
    simple_crop((uint8_t*)sSavedBitmap, BOX_W, BOX_H, (uint8_t*)sSavedBitmap, &out_w, &out_h);
    printf("TP_Save: Cropped to %dx%d\r\n", out_w, out_h);

    // Apply edge detection and save morphological bitmap
    printf("TP_Save: Allocating memory for edge detection (%u bytes)\r\n", 
           (unsigned)(out_w * out_h));

    uint8_t* edges = (uint8_t*)malloc(out_w * out_h);
    if (!edges) {
        printf("TP_Save: ERROR - Failed to allocate %u bytes for edge detection\r\n",
               (unsigned)(out_w * out_h));
        goto user_feedback;
    }

    printf("TP_Save: Running edge detection...\r\n");
    edge_detection((uint8_t*)sSavedBitmap, out_w, out_h, edges);

    #ifdef DEBUG_PRINT
    int edge_count = 0;
    for (int i = 0; i < out_w * out_h; i++) {
        if (edges[i]) edge_count++;
    }
    printf("TP_Save: Edge detection found %d edge pixels\r\n", edge_count);
    #endif

    char filename_morph[64];
    snprintf(filename_morph, sizeof(filename_morph), "/%s.morph.bim", uuid);

    bool res_morph = sd_write_async_packed(edges, out_w, out_h, filename_morph);
    if (res_morph) {
        printf("TP_Save: Queued morphological bitmap write to '%s'\r\n", filename_morph);
    } else {
        printf("TP_Save: Failed to queue morphological bitmap write\r\n");
    }

    free(edges);
    edges = NULL;

    // Feature Extraction 
    printf("\r\n");
    printf("========================================\r\n");
    printf("FEATURE EXTRACTION - UUID: %s\r\n", uuid);
    printf("========================================\r\n");

    // 1. Image Dimensions
    printf("Image Dimensions:\r\n");
    printf("  Width  : %d pixels\r\n", out_w);
    printf("  Height : %d pixels\r\n", out_h);
    printf("  Total  : %d pixels\r\n", out_w * out_h);

    // 2. Connected Components
    printf("\r\nConnected Components:\r\n");
    int n_components = 0;
    connected_components((uint8_t*)sSavedBitmap, out_w, out_h, &n_components);
    printf("  Count  : %d component(s)\r\n", n_components);

    // 3. Pixel Density
    printf("\r\nPixel Density:\r\n");
    float pixel_density = 0.0f;
    density((uint8_t*)sSavedBitmap, out_w, out_h, &pixel_density);
    printf("  Density: %.6f (%.2f%%)\r\n", pixel_density, pixel_density * 100.0f);

    // 4. Edge Count (already computed above)
    printf("\r\nEdge Analysis:\r\n");
    printf("  Edges  : %d pixels\r\n", edge_count);
    float edge_ratio = (out_w * out_h > 0) ? (float)edge_count / (out_w * out_h) : 0.0f;
    printf("  Ratio  : %.6f (%.2f%%)\r\n", edge_ratio, edge_ratio * 100.0f);

    // 5. Compactness Features
    printf("\r\nCompactness Features:\r\n");
    float isoperimetric = 0.0f;
    float a_to_p_ratio = 0.0f;
    float circularity_ratio = 0.0f;
    compactness((uint8_t*)sSavedBitmap, out_w, out_h, 
                &isoperimetric, &a_to_p_ratio, &circularity_ratio);
    
    printf("  Isoperimetric Quotient   : %.6f\r\n", isoperimetric);
    printf("  Area/Perimeter Ratio     : %.6f\r\n", a_to_p_ratio);
    printf("  Circularity Ratio        : %.6f\r\n", circularity_ratio);
    
    // Shape interpretation
    printf("\r\nShape Interpretation:\r\n");
    if (isoperimetric > 0.8f) {
        printf("  Shape: Nearly circular (isoperimetric > 0.8)\r\n");
    } else if (isoperimetric > 0.5f) {
        printf("  Shape: Moderately compact (isoperimetric > 0.5)\r\n");
    } else if (isoperimetric > 0.2f) {
        printf("  Shape: Elongated (isoperimetric > 0.2)\r\n");
    } else {
        printf("  Shape: Very irregular/complex\r\n");
    }

    printf("========================================\r\n");
    printf("END FEATURE EXTRACTION\r\n");
    printf("========================================\r\n\r\n");

    printf("TP_Save: Pipeline complete for UUID %s\r\n", uuid);

user_feedback:
    GUI_DisString_EN(sLCD_DIS.LCD_Dis_Column - 120, 24,
                     "SAVED", &Font16, BLACK, WHITE);
    
    #ifdef DEBUG_PRINT
    printf("TP_Save: Original bitmap dump:\r\n");
    TP_DumpBitmapToSerial(sSavedBitmap);
    #endif
}

/*
This function draws the input of the user on the screen
and prints it on the serial
*/
void TP_DrawBoard(void)
{
    //	sTP_DEV.chStatus &= ~(1 << 6);
    TP_Scan(0);
    if (sTP_DEV.chStatus & TP_PRESS_DOWN)
    { 
        spi_init(SPI_PORT, 10000000);

        // printf("horizontal x:%d,y:%d\n", sTP_Draw.Xpoint, sTP_Draw.Ypoint);

        if (sTP_Draw.Xpoint > (sLCD_DIS.LCD_Dis_Column - 60) &&
            sTP_Draw.Ypoint < 16)
        { 
            TP_Dialog();
        }
        else if (sTP_Draw.Xpoint > (sLCD_DIS.LCD_Dis_Column - 120) &&
                    sTP_Draw.Xpoint < (sLCD_DIS.LCD_Dis_Column - 80) &&
                    sTP_Draw.Ypoint < 24)
        { 
            TP_Save();
        }

        sTP_Draw.Color = BLACK;


        if (sTP_Draw.Xpoint > 100 && sTP_Draw.Xpoint < 380 &&
            sTP_Draw.Ypoint > 50  && sTP_Draw.Ypoint < 290)
        {
            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint, sTP_Draw.Ypoint);

            GUI_DrawPoint(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint);

            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint + 1,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint, sTP_Draw.Ypoint + 1);

            GUI_DrawPoint(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint + 1,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint + 1);

            // The DOT_PIXEL_2X2 point covers the same area; no extra capture needed
            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_2X2, DOT_FILL_RIGHTUP);
        }

        
        spi_init(SPI_PORT, 5000000);
    }
    SPI4W_Write_Byte(0xFF);
}

void TP_Init(LCD_SCAN_DIR Lcd_ScanDir)
{
    DEV_Digital_Write(TP_CS_PIN, 1);

    sTP_DEV.TP_Scan_Dir = Lcd_ScanDir;

    TP_Read_ADC_XY(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint);
}
