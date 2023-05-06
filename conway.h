#ifndef _CONWAYH_
#define _CONWAYH_
#include <stdbool.h>

/**
 * Representation of an individual cell
 */
typedef struct cell {
    bool alive;
    bool will_survive;
} cell_t;

#endif