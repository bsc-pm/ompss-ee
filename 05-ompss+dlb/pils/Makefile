PROGRAM = mpi_ompss_pils
PREFIX = .

TARGETS = $(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION = -mpi

BASE_DIR = ../..
include $(BASE_DIR)/common-files/Makefile

CC = smpcc

CFLAGS = --ompss -cc=$(CC)
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = -lm -mt_mpi
INCS  =

EXTRA = -std=c99 -Wall -Wno-unused

RM = rm -f

SOURCES = mpi_ompss_pils.c

all: $(TARGETS)

$(PROGRAM)-p: $(SOURCES)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-i: $(SOURCES)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-d: $(SOURCES)
	$(MPICC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS)
