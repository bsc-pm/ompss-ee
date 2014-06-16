PROGRAM=array_sum

TARGETS=$(PROGRAM)-p $(PROGRAM)-i $(PROGRAM)-d

BASE_DIR=../..
include $(BASE_DIR)/common-files/Makefile

FC = mfc

FFLAGS = --ompss --no-copy-deps
FFLAGS_P =
FFLAGS_I = --instrument
FFLAGS_D = --debug

LIBS  =
INCS  =

EXTRA = -O3

$(PROGRAM)-p: $(PROGRAM).f90
	$(FC) $(FFLAGS) $(FFLAGS_P) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-i: $(PROGRAM).f90
	$(FC) $(FFLAGS) $(FFLAGS_I) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)-d: $(PROGRAM).f90
	$(FC) $(FFLAGS) $(FFLAGS_D) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(FC)_* *.o *~ $(TARGETS)

