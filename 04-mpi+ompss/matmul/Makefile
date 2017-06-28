PROGRAM=matmul
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-mpi

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = -L$(ATLAS_LIB_DIR) -lcblas -latlas -L$(MPI_LIB_DIR) -lopen-pal -lopen-rte -lmpi
INCS  = -I. -I$(MPI_INC_DIR) -I$(ATLAS_INC_DIR)


EXTRA = -std=c99 -O3 -Wall -Wno-unused 
OBJECTS = driver.o check.o prthead.o gendat.o prtspeed.o cclock.o layouts.o

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS)

.c.o:
	$(CC) --no-openmp $(EXTRA) $(INCS) -c $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

