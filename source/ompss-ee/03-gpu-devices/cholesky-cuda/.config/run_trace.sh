#!/bin/sh
#@ wall_clock_limit = 00:20:00
#@ initialdir = . 
#@ error = cholesky_%j.err
#@ output = cholesky_%j.out
#@ total_tasks = 1
#@ cpus_per_task = 12
#@ gpus_per_node = 2



export NX_INSTRUMENTATION=extrae 
export LD_LIBRARY_PATH=/opt/compilers/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH
export EXTRAE_CONFIG_FILE=../../extrae.xml

for gpus in 1 2 ; do
echo "Number of gpus = $gpus" 
export NX_GPUS=$gpus 
./cholesky_hyb 16384 2048 0
echo " "
done

