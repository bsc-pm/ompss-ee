PROGRAM=heatmpi-ompss-p

export NX_THREADS=2

mpirun ./$PROGRAM test.dat test.ppm
