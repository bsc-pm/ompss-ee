PROGRAM=multisort
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
INCS  = 

EXTRA = -std=c99 -O3 -Wall -Wno-unused 

RM = rm -f

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c
	$(CC) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c
	$(CC) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c
	$(CC) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) 
