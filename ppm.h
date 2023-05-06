

#ifndef _PPMH_
#define _PPMH_
#include <stdbool.h>
#include <string.h>

/**
 * Save generation as a .ppm file
 */
void save_ppm(const char *file_name, int width, int height, cell_t* cell_data)
{
    FILE *fp = fopen(file_name, "wb");
    int max_color = 255;  // maximum color value

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, max_color);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char r;
            unsigned char g;
            unsigned char b;
            if (cell_data[i*width+j].alive) {
                r = 255;
                g = 255;
                b = 255;
            } else {
                r = 0;
                g = 0;
                b = 0;
            }

            fwrite(&r, 1, 1, fp);
            fwrite(&g, 1, 1, fp);
            fwrite(&b, 1, 1, fp);
        }
    }

    // Close file
    fclose(fp);
}

#endif