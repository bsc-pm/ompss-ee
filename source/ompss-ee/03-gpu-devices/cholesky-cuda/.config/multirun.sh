PROGRAM=cholesky_hyb-p

export NX_THREADS=1

export NX_GPUMAXMEM=90

# Executing the application
for gpus in 1 2 ; do
	export NX_GPUS=$gpus 
	./$PROGRAM 16384 2048 0
done

