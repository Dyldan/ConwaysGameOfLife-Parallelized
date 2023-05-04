#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

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
// int count_alive_neighbors(cell_t *cells, int index)
// {
//     int x = index%width;
//     int y = index/width;
//     int count = 0;
//     int north_west = (y-1) * width + (x-1);
//     int north = (y-1) * width + x;
//     int north_east = (y-1) * width + (x+1);
//     int west = index - 1;
//     int east = index + 1;
//     int south_west = (y+1) * width + (x-1);
//     int south = (y+1) * width + x;
//     int south_east = (y-+1) * width + (x+1);

//     if (y > 0) { // NORTH
//         count++;
//     } 
//     if (y < height-1) { // SOUTH
//         count++;
//     } 
//     if (x > 0) { // EAST
//         count++;
//     } 
//     if (x < width - 1) { // WEST
//         count++;
//     } 
//     if (x > 0 && y > 0) { // NEAST
//         count++;
//     } 
//     if (x < width - 1 && y > 0) { // NWEST
//         count++;
//     } 
//     if (x > 0 && y < height-1) { // SEAST
//         count++;
//     } 
//     if (x < width - 1 && y < height-1) { // SWEST
//         count++;
//     } 

//     return count;
// }

/**
 * Spawn the next generation based on the 3 (4 unsimplified) rules.
 * RULES:
 * 1. Any live cell with two or three live neighbours survives.
 * 2. Any dead cell with three live neighbours becomes a live cell.
 * 3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.
 */
__global__
void next_generation(cell_t *cells, int width, int height) // TODO have it return error checking code
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // calculate next generation
    for (int i = index; i < width * height; i += stride) {
        int x = i%width;
        int y = i/width;
        int count = 0;
        int north_west = (y-1) * width + (x-1);
        int north = (y-1) * width + x;
        int north_east = (y-1) * width + (x+1);
        int west = i - 1;
        int east = i + 1;
        int south_west = (y+1) * width + (x-1);
        int south = (y+1) * width + x;
        int south_east = (y-+1) * width + (x+1);

        if (y > 0 && cells[north].alive) { // NORTH
            count++;
        } 
        if (y < height-1 && cells[south].alive) { // SOUTH
            count++;
        } 
        if (x > 0 && cells[east].alive) { // EAST
            count++;
        } 
        if (x < width - 1 && cells[west].alive) { // WEST
            count++;
        } 
        if (x > 0 && y > 0 && cells[north_east].alive) { // NEAST
            count++;
        } 
        if (x < width - 1 && y > 0 && cells[north_west].alive) { // NWEST
            count++;
        } 
        if (x > 0 && y < height-1 && cells[south_east].alive) { // SEAST
            count++;
        } 
        if (x < width - 1 && y < height-1 && cells[south_west].alive) { // SWEST
            count++;
        } 
        int num_alive_neighbors = count;

        if (cells[i].alive && num_alive_neighbors == 2 || num_alive_neighbors == 3) {   // RULE 1
            cells[i].will_survive = true;
        } else if (!cells[i].alive && num_alive_neighbors == 3) { // RULE 2
            cells[i].will_survive = true;
        } else { // RULE 3
            cells[i].will_survive = false;
        }
    }

    // construct next generation
    for (int i = 0; i < width * height; i++) {
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
void construct_starting_cond(cell_t *cells, int cell_count)
{
    for (int i = 0; i < cell_count; i++) {
        if (rand() % cell_count < 1000) {
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
    cudaMallocManaged(&times, sizeof(time_t)*step_count);

    int blockSize = 1024;
    int blockCount = (cell_count + blockSize - 1) / blockSize;
    construct_starting_cond(cells, cell_count);

    START_TIMER(generate)
    next_generation<<<blockCount, blockSize>>>(cells, width, height);
    STOP_TIMER(generate)
    times[0] = GET_TIMER(generate);

    for (int step = 1; step < step_count; step++) {
        to_ppm(cells, step);
        START_TIMER(generate)
        next_generation<<<blockCount, blockSize>>>(cells, width, height);
        STOP_TIMER(generate)
        times[step] = GET_TIMER(generate);
    }

    for (int i = 0; i < step_count-1; i++) {
        printf("%f, ", times[i]);
    }
    printf("%f\n", times[step_count-1]);
    cudaFree(times);
    cudaFree(cells);
    return EXIT_SUCCESS;
}