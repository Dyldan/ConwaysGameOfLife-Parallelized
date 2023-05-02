#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "conway.h"
#include "ppm.h"

#define CELL_COUNT  10000
#define WIDTH       100
#define HEIGHT      100
#define MAX_STEPS   100

cell_t cells[CELL_COUNT];

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
    for (int i = 0; i < CELL_COUNT; i++) {
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
    for (int i = 0; i < CELL_COUNT; i++) {
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
    snprintf(filename, 64, "output/step-%04d.ppm", step);
    save_ppm(filename, WIDTH, HEIGHT, cells);
}

/**
 * Construct the starting grid with ~25% chance of a given cell being
 * alive or dead. Then it invokes all 3 rules on the initial cells.
 */
void construct_starting_cond()
{
    for (int i = 0; i < CELL_COUNT; i++) {
        if (rand() % CELL_COUNT < 1000) {
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
    if (argc > 2) {
        printf("Usage: %s [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int step_count = MAX_STEPS;
    if (argc == 2) {
        step_count = strtol(argv[1], NULL, 10);
    }
    time_t *times;
    times = calloc(step_count, sizeof(time_t));

    construct_starting_cond();
    clock_t start_time = clock();
    next_generation();
    clock_t end_time = clock();
    times[0] = end_time - start_time;

    for (int step = 1; step < step_count; step++) {
        to_ppm(step);
        clock_t start_time = clock();
        next_generation();
        clock_t end_time = clock();
        times[step] = end_time - start_time;
        step++;
    }

    for (int i = 0; i < step_count-1; i++) {
        printf("%ld ms, ", times[i]);
    }
    printf("%ld ms\n", times[step_count-1]);
    free(times);
    return EXIT_SUCCESS;
}