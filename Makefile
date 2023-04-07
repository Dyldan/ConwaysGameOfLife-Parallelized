CC=gcc
CFLAGS=-g -Wall --std=c99 -Wno-unknown-pragmas -O3
NFLAGS=-ccbin $(CC) -g -O3
LIB=-lm
TARGETS=conway_cuda

all: $(TARGETS)

# greyblur_serial: greyblur.c
# 	$(CC) $(CFLAGS) -o $@ $< $(LIB)

conway_cuda: conway.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

clean:
	rm -f $(TARGETS)
	rm -f *.bmp
