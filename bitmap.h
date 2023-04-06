#ifndef _BITMAPH_
#define _BITMAPH_
#include <stdbool.h>
#include <string.h>

void save_bitmap(const char *file_name, int size, int width, int height, cell_t* cell_data)
{
    FILE *image;
    int image_size = size;
    int file_size = image_size + 54 + 4;
    

    struct bitmap_file_header {
    unsigned char   bitmap_type[2];     // 2 bytes
    int             file_size;          // 4 bytes
    short           reserved1;          // 2 bytes
    short           reserved2;          // 2 bytes
    unsigned int    offset_bits;        // 4 bytes
    } bfh;

    struct bitmap_image_header {
        unsigned int    size_header;        // 4 bytes
        unsigned int    width;              // 4 bytes
        unsigned int    height;             // 4 bytes
        short int       planes;             // 2 bytes
        short int       bit_count;          // 2 bytes
        unsigned int    compression;        // 4 bytes
        unsigned int    image_size;         // 4 bytes
        unsigned int    ppm_x;              // 4 bytes
        unsigned int    ppm_y;              // 4 bytes
        unsigned int    clr_used;           // 4 bytes
        unsigned int    clr_important;      // 4 bytes
    } bih;

    memcpy(&(bfh.bitmap_type), "BM", 2);
    bfh.file_size          = file_size;
    bfh.reserved1          = 0;
    bfh.reserved2          = 0;
    bfh.offset_bits        = 0;

    bih.size_header         = sizeof(bih);
    bih.width               = width;
    bih.height              = height;
    bih.planes               = 1;
    bih.bit_count            = 24;
    bih.compression          = 0;
    bih.image_size           = file_size;
    bih.ppm_x                = width;
    bih.ppm_y                = height;
    bih.clr_used             = 0;
    bih.clr_important        = 0;

    image = fopen(file_name, "wb");

    fwrite(&bfh, 1, 14, image);
    fwrite(&bih, 1, sizeof(bih), image);

    for (int i = 0; i < image_size; i++) {
        if (cell_data[i].alive) {
            unsigned char color[3] = {
                255, 255, 255
            };
            fwrite(color, 1, sizeof(color), image);
        } else {
            unsigned char color[3] = {
                0, 0, 0
            };
            fwrite(color, 1, sizeof(color), image);
        }
    }
    fclose(image);
}

#endif