PROGRAM=krist
KERNEL=kernel
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = 
INCS  = -I$(PREFIX)

EXTRA = -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PROGRAM).c $(KERNEL).cl clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl clocks.o

$(PROGRAM)-i: $(PROGRAM).c $(KERNEL).cl clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl clocks.o

$(PROGRAM)-d: $(PROGRAM).c $(KERNEL).cl clocks.o
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl clocks.o

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

