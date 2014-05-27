PROGRAM=krist
KERNEL=kernel
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc
KC = nvcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

NVCFLAGS = -O3

LIBS  = 
INCS  =

EXTRA = -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PROGRAM)-p.o $(KERNEL).o clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) -o $@ $< $(LIBS)

$(PROGRAM)-i: $(PROGRAM)-i.o $(KERNEL).o clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) -o $@ $< $(LIBS)

$(PROGRAM)-d: $(PROGRAM)-d.o $(KERNEL).o clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) -o $@ $< $(LIBS)


$(PROGRAM)-p.o: $(PREFIX)/$(PROGRAM).c krist.h
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-i.o: $(PREFIX)/$(PROGRAM).c krist.h
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-d.o: $(PREFIX)/$(PROGRAM).c krist.h
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<


$(KERNEL).o: $(PREFIX)/$(KERNEL).cu krist.h
	$(KC) -c -o $@ $< $(LIBS)

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

