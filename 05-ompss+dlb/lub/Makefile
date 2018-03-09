PROGRAM = LUB
PREFIX = .

TARGETS = $(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION = -mpi

BASE_DIR = ../..
include $(BASE_DIR)/common-files/Makefile

CC = smpcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug -DCHECK_RESULT

LIBS  = -lm
INCS  =

EXTRA =

RM = rm -f

SOURCES = LUB.c

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
	$(RM) $(CC)_* *.o *~ $(TARGETS)
