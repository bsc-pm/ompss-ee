#PROGRAM=heat-p
# Uncomment to instrument
PROGRAM=heat-i

# Run with 2 threads per MPI process in the same node

export OMP_NUM_THREADS=2

# Uncomment to instrument

$INST ./$PROGRAM test512-jacobi-small.dat test512-jacobi-16.ppm
