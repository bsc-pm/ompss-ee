COMMON_DIR=$(BASE_DIR)/common-files
CONFIG_DIR=.config

GFLAGS=

PARAVER=extrae.xml trace.sh
GRAPH=graph.sh
SCRIPTS=run-once.sh multirun.sh

MPI_CHECK=$(MPI_LIB_DIR)/libmpi.so
MKL_CHECK=$(MKL_LIB_DIR)/libmkl_sequential.so
ATLAS_CHECK=$(ATLAS_LIB_DIR)/libatlas.a

all: $(TARGETS) $(SCRIPTS) $(PARAVER) $(GRAPH)

extrae.xml:
	cp $(COMMON_DIR)/extrae.xml .

trace.sh:
	cp $(COMMON_DIR)/trace$(TRACE_VERSION).sh trace.sh

graph.sh:
	cp $(COMMON_DIR)/graph.sh .

run-once.sh: $(COMMON_DIR)/run-once.sh $(CONFIG_DIR)/run-once.sh
	cp  $(COMMON_DIR)/run-once.sh .
	cat $(COMMON_DIR)/sched-job$(JOB_SCHED_VERSION) >> run-once.sh
	cat $(CONFIG_DIR)/run-once.sh >> run-once.sh

multirun.sh: $(COMMON_DIR)/multirun.sh $(CONFIG_DIR)/multirun.sh
	cp  $(COMMON_DIR)/multirun.sh .
	cat $(COMMON_DIR)/sched-job$(JOB_SCHED_VERSION) >> multirun.sh
	cat $(CONFIG_DIR)/multirun.sh >> multirun.sh

$(MPI_CHECK):
	@echo "=================================================="
	@echo "===           MPI LIBRARY NEEDED               ==="
	@echo "=================================================="
	@false

$(MKL_CHECK):
	@echo "=================================================="
	@echo "===           MKL LIBRARY NEEDED               ==="
	@echo "=================================================="
	@false

$(ATLAS_CHECK):
	@echo "=================================================="
	@echo "===          ATLAS LIBRARY NEEDED              ==="
	@echo "=================================================="
	@false

wipe: clean
	rm -f $(PARAVER)
	rm -f $(SCRIPTS)
	rm -f *.out *.err
	rm -f *.prv *.row *.pcf
	rm -f TRACE.mpits
	rm -f -r set-0
	rm -f graph.dot graph.pdf
