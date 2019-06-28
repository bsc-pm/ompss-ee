PROGRAM=matmul-p

THREADS=(1 2 3 4 5 6)

for thread in ${THREADS[@]}; do
	NX_SMP_WORKERS=$thread ${MPIRUN_COMMAND} ./$PROGRAM
done
