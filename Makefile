CC=gcc
CFLAGS=-g -Wall --std=c99 -Wno-unknown-pragmas -O3
NFLAGS=-ccbin $(CC) -g -O3
LIB=-lm
TARGETS=conway_ser conway_cuda

all: $(TARGETS)



conway_ser: conway.c
	gcc -o conway_ser conway.c

conway_cuda: conway.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

convert: output/*.ppm
	convert -delay 10 -loop 1 output/*.ppm animation.gif

clean:
	rm -f $(TARGETS)
	rm -f output/*.ppm
	rm -f animation.gif
	rm -f serial_timings/*
	rm -f parallel_timings/*
