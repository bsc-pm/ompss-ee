PROGRAM=dot-product
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-smp

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc

CFLAGS = --ompss --no-copy-deps
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = 
INCS  =

EXTRA = -std=c99 -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

