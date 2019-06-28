PROGRAM=krist
KERNEL=kernel
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-smp

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc
KC = nvcc

CFLAGS = --ompss --cuda
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

NVCFLAGS = -O3

LIBS  = 
INCS  = -I$(PREFIX)

EXTRA = -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PROGRAM)-p.o $(KERNEL).o clocks.o
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o clocks.o

$(PROGRAM)-i: $(PROGRAM)-i.o $(KERNEL).o clocks.o
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o clocks.o

$(PROGRAM)-d: $(PROGRAM)-d.o $(KERNEL).o clocks.o
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o clocks.o


$(PROGRAM)-p.o: $(PROGRAM).c $(PREFIX)/krist.h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-i.o: $(PROGRAM).c $(PREFIX)/krist.h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-d.o: $(PROGRAM).c $(PREFIX)/krist.h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<


$(KERNEL).o: $(KERNEL).cu $(PREFIX)/krist.h
	$(KC) $(INCS) -o $@ -c $< 

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

