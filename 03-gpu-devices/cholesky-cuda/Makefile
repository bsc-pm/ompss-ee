PROGRAM=cholesky_hyb
KERNEL=cuda_potrf
PREFIX=.

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-smp

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc
KC = nvcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

NVCFLAGS = -O3 -arch=sm_20


LIBS  = --Wl,-L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_rt -lpthread
INCS  = -I$(PREFIX) -I$(MKL_INC_DIR)

EXTRA = -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PROGRAM)-p.o $(KERNEL).o $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o

$(PROGRAM)-i: $(PROGRAM)-i.o $(KERNEL).o $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o

$(PROGRAM)-d: $(PROGRAM)-d.o $(KERNEL).o $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) -o $@ $< $(LIBS) $(KERNEL).o


$(PROGRAM)-p.o: $(PREFIX)/$(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-i.o: $(PREFIX)/$(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ -c $<

$(PROGRAM)-d.o: $(PREFIX)/$(PROGRAM).c $(PREFIX)/$(KERNEL).h
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ -c $<


$(KERNEL).o: $(KERNEL).cu $(PREFIX)/$(KERNEL).h
	$(KC) $(NVCFLAGS) $(INCS) -o $@ -c $< 

.c.o:
	$(CC) --no-openmp $(EXTRA) -c $<

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

