#!/bin/bash

# Uncomment following line to trace MPI programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libnanosmpitrace.so

export EXTRAE_CONFIG_FILE=extrae.xml
export NX_INSTRUMENTATION=extrae

$*
