PROGRAM=heat-mpi
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

# Use MPI sched job
EXTRA_SCHED_OPTIONS=-mpi

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

EXTRAEI = -I$(EXTRAE_HOME)/include/ -I.
EXTRAEL = -L$(EXTRAE_HOME)/lib -Wl,-rpath=$(EXTRAE_HOME)/lib -lompitrace -lxml2 

CFLAGS = -openmp -qopenmp-lib=compat
CFLAGS = -fopenmp
CFLAGS_P = -O3
CFLAGS_I = -O3 -g  -D_EXTRAE_
CFLAGS_D = -O0 -g2

LIBS  = -lm 
INCS  = 

EXTRA = -std=c99 -Wall -Wno-unused 

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



