PROGRAM=heat-mpi
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

# Use MPI sched job
EXTRA_SCHED_OPTIONS=-mpi

BASE_DIR=../..
include ../../common-files/Makefile

CC = gcc

CFLAGS = -fopenmp
CFLAGS_P = -O2
CFLAGS_I = 
CFLAGS_D = -O0 -g2

LIBS  = -L $(MPI_LIB_DIR) -lpthread -lc -lm -lutil -lnsl -ldl -lopen-pal -lopen-rte -lrt -libverbs -lmpi
INCS  = -I$(MPI_INC_DIR) 

EXTRA = -std=c99 -O3 -Wall -Wno-unused 

RM = rm -f

OBJECTS = misc.c solver-mpi.c

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) test.ppm



