PROGRAM = ompss_pils
PREFIX = .

TARGETS = $(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION = -smp

BASE_DIR = ../..
include $(BASE_DIR)/common-files/Makefile

CC = smpcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  =
INCS  =

EXTRA = -std=c99 -Wall -Wno-unused

RM = rm -f

SOURCES = ompss_pils.c

all: $(TARGETS)

$(PROGRAM)-p: $(SOURCES)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-i: $(SOURCES)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

$(PROGRAM)-d: $(SOURCES)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(CC)_* *.o *~ $(TARGETS)
