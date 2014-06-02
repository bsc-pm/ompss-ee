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

LIBS  = -L/usr/lib64 -L/opt/compilers/intel/lib/intel64/ -lirc -L/opt/mpi/bullxmpi/1.1.11.1/lib -lpthread -lc -lm -lutil -lnsl -ldl -lopen-pal -lopen-rte -lrt -libverbs -lmpi  -L/gpfs/apps/NVIDIA/ATLAS/3.9.51/lib -lcblas -latlas

INCS  = -I. -I/opt/mpi/bullxmpi/1.1.11.1/include -I/gpfs/apps/NVIDIA/ATLAS/3.9.51/include/

EXTRA = -std=c99 -O3 -Wall -Wno-unused 
OBJECTS = misc.o

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-p.o
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-p.o

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-i.o
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-i.o

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS) $(SOLVER)-d.o
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(OBJECTS) $(SOLVER)-d.o

$(SOLVER)-p.o: $(PREFIX)/$(SOLVER).c
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(SOLVER)-i.o: $(PREFIX)/$(SOLVER).c
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(SOLVER)-d.o: $(PREFIX)/$(SOLVER).c
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

