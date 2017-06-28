PROGRAM=cholesky

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

JOB_SCHED_VERSION=-smp

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

CC = mcc

CFLAGS = --ompss
CFLAGS_P =
CFLAGS_I = --instrument
CFLAGS_D = --debug

LIBS  = --Wl,-L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_rt -lpthread
INCS  = -I$(MKL_INC_DIR)

EXTRA = -std=c99 -O3 -Wall -Wno-unused 

$(PROGRAM)-p: $(PROGRAM).c $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-i: $(PROGRAM).c $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-d: $(PROGRAM).c $(MKL_CHECK)
	$(CC) $(GFLAGS) $(CFLAGS) $(CFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

