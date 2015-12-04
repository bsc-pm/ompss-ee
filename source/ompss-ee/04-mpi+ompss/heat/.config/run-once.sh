PROGRAM=heat-mpi-ompss-i

# Run with 2 threads per MPI process in the same node

export SMP_NUM_WORKERS=2

# Uncomment to instrument
#export INST=./graph.sh
#export INST=./trace.sh

mpirun $INST ./$PROGRAM test.dat test.ppm
