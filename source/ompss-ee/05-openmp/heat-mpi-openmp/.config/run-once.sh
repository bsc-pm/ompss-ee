PROGRAM=heat-mpi-i

# Run with 2 threads per MPI process in the same node

export OMP_NUM_THREADS=2

# Uncomment to instrument
#export INST=./graph.sh
#export INST=./trace.sh

mpirun --cpus-per-proc 2 --bind-to-core $INST ./$PROGRAM test.dat test.ppm
