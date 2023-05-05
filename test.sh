#!/bin/bash


# SERIAL GAME AREA
./conway_ser 1000 1000 10 > serial_timings/serial_1k1k10.csv
./conway_ser 2000 2000 10 > serial_timings/serial_2k2k10.csv
./conway_ser 4000 4000 10 > serial_timings/serial_4k4k10.csv
./conway_ser 6000 6000 10 > serial_timings/serial_6k6k10.csv
./conway_ser 8000 8000 10 > serial_timings/serial_8k8k10.csv
./conway_ser 10000 10000 10 > serial_timings/serial_10k10k10.csv

# SERIAL COUNT
./conway_ser 100 100 100 > serial_timings/serial_1k1k1c.csv
./conway_ser 100 100 200 > serial_timings/serial_1k1k2c.csv
./conway_ser 100 100 400 > serial_timings/serial_1k1k4c.csv
./conway_ser 100 100 500 > serial_timings/serial_1k1k5c.csv
./conway_ser 100 100 1000 > serial_timings/serial_1k1k1k.csv


# PARALLEL GAME AREA
./conway_cuda 1000 1000 10 > parallel_timings/parallel_1k1k10.csv
./conway_cuda 2000 2000 10 > parallel_timings/parallel_2k2k10.csv
./conway_cuda 4000 4000 10 > parallel_timings/parallel_4k4k10.csv
./conway_cuda 6000 6000 10 > parallel_timings/parallel_6k6k10.csv
./conway_cuda 8000 8000 10 > parallel_timings/parallel_8k8k10.csv
./conway_cuda 10000 10000 10 > parallel_timings/parallel_10k10k10.csv

# PARALLEL COUNT
./conway_cuda 100 100 100 > parallel_timings/parallel_1k1k1c.csv
./conway_cuda 100 100 200 > parallel_timings/parallel_1k1k2c.csv
./conway_cuda 100 100 400 > parallel_timings/parallel_1k1k4c.csv
./conway_cuda 100 100 500 > parallel_timings/parallel_1k1k5c.csv
./conway_cuda 100 100 1000 > parallel_timings/parallel_1k1k1k.csv




