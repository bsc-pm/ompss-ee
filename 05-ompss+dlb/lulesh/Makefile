PROGRAM = lulesh2.0
PREFIX = .

TARGETS = $(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION = -mpi

BASE_DIR = ../..
include $(BASE_DIR)/common-files/Makefile

CXX = smpcxx

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = -lm
INCS  =

EXTRA = -DUSE_MPI=1

RM = rm -f

SOURCES = lulesh.cc lulesh-comm.cc lulesh-viz.cc lulesh-util.cc lulesh-init.cc

all: $(TARGETS)

$(PROGRAM)-p: $(SOURCES)
	$(MPI_NATIVE_CXX)=$(CXX) \
	$(MPICXX) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-i: $(SOURCES)
	$(MPI_NATIVE_CXX)=$(CXX) \
	$(MPICXX) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-d: $(SOURCES)
	$(MPI_NATIVE_CXX)=$(CXX) \
	$(MPICXX) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS)
