PROGRAM=heat-mpi-ompss
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-mpi

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = smpcc

CFLAGS = --ompss  -cc=$(CC)
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = -lm -mt_mpi
INCS  = 

EXTRA = -std=c99 -Wall -Wno-unused 

RM = rm -f

OBJECTS = misc.c solver-mpi-ompss.c

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) test.ppm



