COMMON_DIR=$(BASE_DIR)/common-files
CONFIG_DIR=.config

PARAVER=extrae.xml trace.sh
SCRIPTS=run-once.sh multirun.sh

MKL_CHECK=$(MKL_LIB_DIR)/libmkl_sequential.so

all: $(TARGETS) $(SCRIPTS) $(PARAVER)

extrae.xml:
	cp $(COMMON_DIR)/extrae.xml . 

trace.sh:
	cp $(COMMON_DIR)/trace.sh . 

run-once.sh:
	cp  $(COMMON_DIR)/run-once.sh . 
	cat $(COMMON_DIR)/sched-job >> run-once.sh 
	cat $(CONFIG_DIR)/run-once.sh >> run-once.sh 

multirun.sh:
	cp  $(COMMON_DIR)/multirun.sh . 
	cat $(COMMON_DIR)/sched-job >> multirun.sh 
	cat $(CONFIG_DIR)/multirun.sh >> multirun.sh 

$(MKL_CHECK):
	@echo "=================================================="
	@echo "===           MKL LIBRARY NEEDED               ==="
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
