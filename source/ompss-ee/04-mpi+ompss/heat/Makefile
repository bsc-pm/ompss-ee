PROGRAM=heat
SOLVER=solver
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

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
OBJECTS = misc.o

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-p.o $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-p.o

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-i.o $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-i.o

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-d.o $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-d.o

$(SOLVER)-p.o: $(PREFIX)/$(SOLVER).c $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(SOLVER)-i.o: $(PREFIX)/$(SOLVER).c $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(SOLVER)-d.o: $(PREFIX)/$(SOLVER).c $(ATLAS_CHECK) $(MPI_CHECK)
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

