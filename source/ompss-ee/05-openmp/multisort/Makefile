PROGRAM=multisort
PREFIX=.

TARGETS=$(PROGRAM)-tree-p $(PROGRAM)-tree-i $(PROGRAM)-tree-d $(PROGRAM)-leaf-p $(PROGRAM)-leaf-i $(PROGRAM)-leaf-d $(PROGRAM)-serial

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

OBJECTS = kernels.c

all: $(TARGETS)

$(PROGRAM)-tree-p: $(PREFIX)/$(PROGRAM)-tree.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-tree-i: $(PREFIX)/$(PROGRAM)-tree.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(EXTRAEI) $(INCS) -o $@ $< $(OBJECTS) $(LIBS) $(EXTRAEL)

$(PROGRAM)-tree-d: $(PREFIX)/$(PROGRAM)-tree.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-leaf-p: $(PREFIX)/$(PROGRAM)-leaf.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-leaf-i: $(PREFIX)/$(PROGRAM)-leaf.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(EXTRAEI) $(INCS) -o $@ $< $(OBJECTS) $(LIBS) $(EXTRAEL)

$(PROGRAM)-leaf-d: $(PREFIX)/$(PROGRAM)-leaf.c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

$(PROGRAM)-serial: $(PREFIX)/$(PROGRAM).c $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(OBJECTS) $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS)




