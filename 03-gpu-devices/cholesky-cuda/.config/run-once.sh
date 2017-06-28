PROGRAM=cholesky_hyb-p

export NX_SMP_WORKERS=1
export NX_GPUS=2 #change this in order to use more GPUs

export NX_GPUMAXMEM=90

# Executing the application
./$PROGRAM 16384 2048 0
