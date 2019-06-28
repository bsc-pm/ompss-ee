PROGRAM=heat-mpi-ompss-p

THREADS=(1 2 3 4 5 6)
INPUT=test.dat

for thread in ${THREADS[@]}; do
  NX_SMP_WORKERS=$thread ${MPIRUN_COMMAND} ./$PROGRAM $INPUT test.ppm
done
