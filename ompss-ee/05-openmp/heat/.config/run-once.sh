#PROGRAM=heat-p
# Uncomment to instrument
PROGRAM=heat-i

# Run with 2 threads
export OMP_NUM_THREADS=2

# Uncomment to instrument
INST=./trace.sh

$INST ./$PROGRAM test512-jacobi-small.dat test512-jacobi-16.ppm
