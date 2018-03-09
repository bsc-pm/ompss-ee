PROGRAM=matmul
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

LIBS  = -L$(ATLAS_LIB_DIR) -lcblas -latlas
INCS  = -I. -I$(ATLAS_INC_DIR)


EXTRA = -std=c99 -O3 -Wall -Wno-unused
SOURCES = matmul.c driver.c check.c prthead.c gendat.c prtspeed.c cclock.c layouts.c

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
	rm -f $(CC)_* *.o *~ $(TARGETS)

