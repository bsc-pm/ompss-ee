PROGRAM=heat
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d $(PROGRAM)-serial

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = icc

CFLAGS = -fopenmp
CFLAGS_P = -O3
CFLAGS_I = -O3 -g
CFLAGS_D = -O0 -g2

EXTRAEI = -I$(EXTRAE_HOME)/include
EXTRAEL = -L$(EXTRAE_HOME)/lib -Wl,-rpath=$(EXTRAE_HOME)/lib -lomptrace

LIBS  = -lm 
INCS  = 

EXTRA = -std=c99 -Wall -Wno-unused 

RM = rm -f

OBJECTS = misc.c solver.c

all: $(TARGETS)

$(PROGRAM)-p: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-i: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(EXTRAEI) $(INCS) -o $@ $< $(OBJECTS) $(LIBS) $(EXTRAEL)

$(PROGRAM)-d: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-serial: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS) *.ppm




