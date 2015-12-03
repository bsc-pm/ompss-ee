PROGRAM=heat-mpi-i

# Run with 2 threads per MPI process in the same node

export OMP_NUM_THREADS=2

# Uncomment to instrument
#export INST=./trace.sh

mpirun $INST ./$PROGRAM test.dat test.ppm
