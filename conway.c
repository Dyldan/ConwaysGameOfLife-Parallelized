#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

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
int count_alive_neighbors(cell_t cell, int index)
{
    int count = 0;

    if (index >= 100 && cells[index - 100].alive) { // NORTH
        count++;
    }

    if (index <= 9899 && cells[index + 100].alive) { // SOUTH
        count++;
    }

    if (index % 100 != 0 && cells[index - 1].alive) { // WEST
        count++;
    }

    if (index % 100 != 99 && cells[index + 1].alive) { // EAST
        count++;
    }

    if (index >= 100 && index % 100 != 0 && cells[index - 101].alive) { // NORTH-WEST
        count++;
    }

    if (index >= 100 && index % 100 != 99 && cells[index - 99].alive) { // NORTH-EAST
        count++;
    }

    if (index <= 9899 && index % 100 != 0 && cells[index + 99].alive) { // SOUTH-WEST
        count++;
    }

    if (index <= 9899 && index % 100 != 99 && cells[index + 101].alive) { // SOUTH-EAST
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
 */
void next_generation() // TODO have it return error checking code
{
    // calculate next generation
    for (int i = 0; i < cell_count; i++) {
        int num_alive_neighbors = count_alive_neighbors(cells[i], i);

        if (cells[i].alive && num_alive_neighbors == 2 || num_alive_neighbors == 3) {   // RULE 1
            cells[i].will_survive = true;
        } else if (!cells[i].alive && num_alive_neighbors == 3) { // RULE 2
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
    snprintf(filename, 64, "output/step-%05d.ppm", step); // TODO make sure output folder exists / create it
    save_ppm(filename, width, height, cells);
}

/**
 * Construct the starting grid with ~25% chance of a given cell being
 * alive or dead. Then it invokes all 3 rules on the initial cells.
 */
void construct_starting_cond()
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
    cells = calloc(cell_count, sizeof(cell_t));

    time_t *times;
    times = calloc(step_count+1, sizeof(time_t));
    clock_t start_time = clock();
    construct_starting_cond();
    clock_t end_time = clock();
    times[0] = end_time - start_time;
    to_ppm(0);

    for (int step = 1; step <= step_count; step++) {
        clock_t start_time = clock();
        next_generation();
        clock_t end_time = clock();
        to_ppm(step);
        times[step] = end_time - start_time;
    }

    for (int i = 0; i <= step_count; i++) {
        if (i == 0) {
            printf("%ld (base), ", times[i]);
        } else if (i == step_count) {
            printf("%ld\n", times[i]);
        } else {
            printf("%ld, ", times[i]);
        }
    }
    free(times);
    free(cells);
    return EXIT_SUCCESS;
}