PROGRAM=saxpy
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
INCS  = -I$(PREFIX)

EXTRA = -O3 -Wall -Wno-unused 
OBJECTS=

$(PROGRAM)-p: $(PROGRAM)-p.o $(KERNEL).o $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o $(OBJECTS)

$(PROGRAM)-i: $(PROGRAM)-i.o $(KERNEL).o $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o $(OBJECTS)

$(PROGRAM)-d: $(PROGRAM)-d.o $(KERNEL).o $(OBJECTS)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o $(OBJECTS)


$(PROGRAM)-p.o: $(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-i.o: $(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-d.o: $(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<


$(KERNEL).o: $(KERNEL).cu $(PREFIX)/$(KERNEL).h
	$(KC) $(INCS) -o $@ -c $< 

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

