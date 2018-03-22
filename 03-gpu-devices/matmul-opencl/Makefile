PROGRAM=matmul
KERNEL=kernel
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-smp

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc

CFLAGS = --ompss --opencl
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = 
INCS  = -I$(PREFIX) --opencl-build-opts=-I$(PREFIX)

EXTRA = -O3 -Wall -Wno-unused 
OBJECTS = cclock.o driver.o prtspeed.o check.o gendat.o

$(PROGRAM)-p: $(PROGRAM).c $(KERNEL).cl $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl $(OBJECTS)

$(PROGRAM)-i: $(PROGRAM).c $(KERNEL).cl $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl $(OBJECTS)

$(PROGRAM)-d: $(PROGRAM).c $(KERNEL).cl $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS) $(KERNEL).cl $(OBJECTS)

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

