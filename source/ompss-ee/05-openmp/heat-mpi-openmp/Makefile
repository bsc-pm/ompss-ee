PROGRAM=heat-mpi
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

# Use MPI sched job
EXTRA_SCHED_OPTIONS=-mpi

BASE_DIR=../..
include ../../common-files/Makefile

EXTRAEI = -I$(EXTRAE_HOME)/include/ -I.
EXTRAEL = -L$(EXTRAE_HOME)/lib -Wl,-rpath=$(EXTRAE_HOME)/lib -lompitrace -lxml2 

CFLAGS = -openmp -qopenmp-lib=compat
CFLAGS_P = -O3
CFLAGS_I = 
CFLAGS_D = -O0 -g2

LIBS  = -lpthread -lc -lm 
INCS  = 

EXTRA = -std=c99 -O3 -Wall -Wno-unused 

RM = rm -f

OBJECTS = misc.c solver-mpi.c

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(EXTRAEI) $(INCS) -o $@ $< $(OBJECTS) $(LIBS) $(EXTRAEL)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) test.ppm



