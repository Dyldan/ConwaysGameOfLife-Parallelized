#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "conway.h"
#include "ppm.h"

#define DEF_STEPS   100

int height;
int width;
int cell_count;
cell_t *cells;

/**
 * Returns the amount of alive cells that neighbor the given cell (max 8).
 */
int count_alive_neighbors(int index)
{
    int count = 0;

    if (index >= width && cells[index - width].alive) { // NORTH
        count++;
    }

    if (index < cell_count - width && cells[index + width].alive) { // SOUTH
        count++;
    }

    if (index % width != 0 && cells[index - 1].alive) { // WEST
        count++;
    }

    if (index % width != width-1 && cells[index + 1].alive) { // EAST
        count++;
    }

    if (index >= width && index % width != 0 && cells[index - (width+1)].alive) { // NORTH-WEST
        count++;
    }

    if (index >= width && index % width != width-1 && cells[index - (width-1)].alive) { // NORTH-EAST
        count++;
    }

    if (index < cell_count - width && index % width != 0 && cells[index + (width-1)].alive) { // SOUTH-WEST
        count++;
    }

    if (index < cell_count - width && index % width != width-1 && cells[index + (width+1)].alive) { // SOUTH-EAST
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
void next_generation()
{
    // calculate next generation
    for (int i = 0; i < cell_count; i++) {
        int num_alive_neighbors = count_alive_neighbors(i);

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
    for (int i = 0; i < cell_count; i++) {
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
void to_ppm(int step) {
    static char filename[64];
    snprintf(filename, 64, "output/step-%05d.ppm", step);
    save_ppm(filename, width, height, cells);
}

/**
 * Construct the starting grid with ~25% chance of a given cell being
 * alive or dead. Then it invokes all 3 rules on the initial cells.
 */
void construct_starting_cond()
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
    // Argument Handling
    if (argc > 4 || argc < 3) {
        printf("Usage: %s <height> <width> [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    height = strtol(argv[1], NULL, 10);
    width = strtol(argv[2], NULL, 10);

    // Init optional step count
    int step_count = DEF_STEPS;
    if (argc == 4) {
        step_count = strtol(argv[3], NULL, 10);
    }

    // Verify command line args
    if (height == 0 || width == 0 || step_count == 0) {
        printf("Usage: %s <height> <width> [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Init variables
    cell_count = height * width;
    cells = calloc(cell_count, sizeof(cell_t));

    double *times;
    times = calloc(step_count+1, sizeof(double));

    double start_time;
    double end_time;

    // Construct genesis generation
    start_time = clock();
    construct_starting_cond();
    end_time = clock();
    times[0] = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    to_ppm(0);

    // Main program loop - construct each generation one at a time
    for (int step = 1; step <= step_count; step++) {
        start_time = clock();
        next_generation();
        end_time = clock();
        times[step] = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        to_ppm(step);
    }

    // Print timing output
    for (int i = 0; i <= step_count; i++) {
        if (i == 0) {
            printf("%f\n", times[i]);
        } else if (i == step_count) {
            printf("%f\n", times[i]);
        } else {
            printf("%f, ", times[i]);
        }
    }

    // Clean up and exit
    free(times);
    free(cells);
    return EXIT_SUCCESS;
}