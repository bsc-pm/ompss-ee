#!/bin/bash

# Uncomment the following line to trace MPI+OmpSs programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libnanosmpitrace.so

# Uncomment the following line to trace MPI+OpenMP (GNU) programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitrace.so

export EXTRAE_CONFIG_FILE=extrae.xml
export NX_INSTRUMENTATION=extrae

$*

mpi2prv -f TRACE.mpits -o myTrace.prv
