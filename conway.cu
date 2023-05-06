#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#include "conway.h"
#include "ppm.h"
#include "timer.h"

#define DEF_STEPS   100

int height;
int width;
int cell_count;

/**
 * Returns the amount of alive cells that neighbor the given cell (max 8).
 */
__device__
int count_alive_neighbors(int width, int height, cell_t *cells, int i)
{
    int count = 0;
    
    if (i >= width && cells[i - width].alive) { // NORTH
        count++;
    }

    if (i < (width * height) - width && cells[i + width].alive) { // SOUTH
        count++;
    }

    if (i % width != 0 && cells[i - 1].alive) { // WEST
        count++;
    }

    if (i % width != width-1 && cells[i + 1].alive) { // EAST
        count++;
    }

    if (i >= width && i % width != 0 && cells[i - (width+1)].alive) { // NORTH-WEST
        count++;
    }

    if (i >= width && i % width != width-1 && cells[i - (width-1)].alive) { // NORTH-EAST
        count++;
    }

    if (i < (width * height) - width && i % width != 0 && cells[i + (width-1)].alive) { // SOUTH-WEST
        count++;
    }

    if (i < (width * height) - width && i % width != width-1 && cells[i + (width+1)].alive) { // SOUTH-EAST
        count++;
    }

    return count;
}

/**
 * Spawn the next generation based on the 3 (4 unsimplified) rules.
 * RULES:
 * 1. Any live cell with two or three live neighbours survives.
 * 2. Any dead cell with three live neighbours becomes a live cell.
 * 3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.
 * 4. [CUSTOM] Any live cell with 5 live neighbors survives.
 * 5. [CUSTOM] Any dead cell with 5 live neighbors becomes a live cell.
 */
__global__
void next_generation(cell_t *cells, int width, int height) // TODO have it return error checking code
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // calculate next generation
    for (int i = index; i < width * height; i += stride) {
        int num_alive_neighbors = count_alive_neighbors(width, height, cells, i);

        if (cells[i].alive && num_alive_neighbors == 2 || num_alive_neighbors == 3) {   // RULE 1
            cells[i].will_survive = true;
        } else if (!cells[i].alive && num_alive_neighbors == 3) { // RULE 2
            cells[i].will_survive = true;
        } else if (cells[i].alive && num_alive_neighbors == 5) { // CUSTOM RULE 4
            cells[i].will_survive = true;
        } else if (!cells[i].alive && num_alive_neighbors == 5) { // CUSTOME RULE 5
            cells[i].will_survive = true;
        } else { // RULE 3
            cells[i].will_survive = false;
        }
    }

    // construct next generation
    for (int i = index; i < width * height; i+= stride) {
        if (cells[i].will_survive) {
            cells[i].alive = true;
        } else {
            cells[i].alive = false;
        }
    }
}

/**
 * Gets correct name for new image and sends to the save_ppm function
*/
void to_ppm(cell_t *cells, int step) {
    static char filename[64];
    snprintf(filename, 64, "output/step-%05d.ppm", step);
    save_ppm(filename, width, height, cells);
}

/**
 * Construct the starting grid with ~25% chance of a given cell being
 * alive or dead. Then it invokes all 3 rules on the initial cells.
 */
void construct_starting_cond(cell_t *cells)
{
    struct stat st = {0};
    if (stat("output/", &st) == -1 ) {
        mkdir("output/", 0700);
    }
    srand(time(NULL));
    for (int i = 0; i < cell_count; i++) {
        if (rand() % cell_count < (width*height) / 10) {
            cells[i].alive = true;
        } else {
            cells[i].alive = false;
        }
    }
}

/**
 * Main.
 */
int main(int argc, char *argv[])
{
    if (argc > 4 || argc < 3) {
        printf("Usage: %s <height> <width> [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    height = strtol(argv[1], NULL, 10);
    width = strtol(argv[2], NULL, 10);

    int step_count = DEF_STEPS;
    if (argc == 4) {
        step_count = strtol(argv[3], NULL, 10);
    }

    if (height == 0 || width == 0 || step_count == 0) {
        printf("Usage: %s <height> <width> [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    cell_count = height * width;
    cell_t *cells;
    cudaMallocManaged(&cells, sizeof(cell_t)*cell_count);

    float *times;
    cudaMallocManaged(&times, sizeof(time_t)*(step_count+1));

    int blockSize = 1024;
    int blockCount = (cell_count + blockSize - 1) / blockSize;

    START_TIMER(generate)
    construct_starting_cond(cells);
    STOP_TIMER(generate)
    times[0] = GET_TIMER(generate);
    to_ppm(cells, 0);

    for (int step = 1; step <= step_count; step++) {
        START_TIMER(generate)
        next_generation<<<blockCount, blockSize>>>(cells, width, height);
        STOP_TIMER(generate)
        times[step] = GET_TIMER(generate);
        to_ppm(cells, step);
    }

    for (int i = 0; i <= step_count; i++) {
        if (i == 0) {
            printf("%f\n", times[i]);
        } else if (i == step_count) {
            printf("%f\n", times[i]);
        } else {
            printf("%f, ", times[i]);
        }
    }
    cudaFree(times);
    cudaFree(cells);
    return EXIT_SUCCESS;
}