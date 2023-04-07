#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "conway.h"
#include "bitmap.h"

#define CELL_COUNT 10000
#define WIDTH      100
#define HEIGHT     100

cell_t cells[CELL_COUNT];

/**
 * Returns the amount of alive cells that neighbor the given cell (max 8).
 */
int count_alive_neighbors(cell_t cell, int index) // TODO account for walls of grid
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
    for (int i = 0; i < CELL_COUNT; i++) {
        int num_alive_neighbors = count_alive_neighbors(cells[i], i);
      
        if (cells[i].alive && num_alive_neighbors == 2 || num_alive_neighbors == 3) {   // RULE 1
            cells[i].will_survive = true;
        } else if (!cells[i].alive && num_alive_neighbors == 3) { // RULE 2
            cells[i].will_survive = true;
        } else if (cells[i].alive) { // RULE 3
            cells[i].will_survive = false;
        }
    }

    for (int i = 0; i < CELL_COUNT; i++) {
        if (cells[i].will_survive) {
            cells[i].alive = true;
        }
    }
}

/**
 * Construct the starting grid with ~50% chance of a given cell being
 * alive or dead. Then it invokes all 3 rules on the initial cells.
 */
void construct_starting_cond()
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < CELL_COUNT; i++) {
        if (rand() % CELL_COUNT < 5000) {
            cells[i].alive = true;
        } else {
            cells[i].alive = false;
        }
    }

    next_generation();
}

/**
 * Gets correct name for new image and sends to the save_bitmap function
*/
void to_bitmap(int step) {
    static char filename[32];
    snprintf(filename, 16, "step-%d.bmp", step);
    save_bitmap(filename, CELL_COUNT, WIDTH, HEIGHT, cells);
}

/**
 * Main.
 */
int main()
{
    construct_starting_cond();
    int step = 0;
    while (true) {
        // for (int i = 0; i < CELL_COUNT; i++) {
        //     if (i % 100 == 0) {
        //         printf("\n");
        //     }
        //     if (cells[i].alive) {
        //         printf("x");
        //     } else {
        //         printf(" ");
        //     }
        // }
        // printf("\n\n\n\n\n");
        int count = 0;
        for (int i = 0; i< CELL_COUNT; i++) {
            if (cells[i].alive) {
                count++;
            }
        }
        to_bitmap(step);
        printf("%d\n", count);
        sleep(1);
        next_generation();
        step++;
    }

    return EXIT_SUCCESS;
}