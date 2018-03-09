PROGRAM=heat-mpi-ompss
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-mpi

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = smpcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = -lm
INCS  = 

EXTRA = -std=c99 -Wall -Wno-unused 

RM = rm -f

SOURCES = heat-mpi-ompss.c misc.c solver-mpi-ompss.c

all: $(TARGETS)

$(PROGRAM)-p: $(SOURCES)
	$(MPI_NATIVE_CC)=$(CC) \
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-i: $(SOURCES)
	$(MPI_NATIVE_CC)=$(CC) \
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-d: $(SOURCES)
	$(MPI_NATIVE_CC)=$(CC) \
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) test.ppm



