PROGRAM=krist-p

export NX_GPUMAXMEM=150000000
export NX_GPUS=1 #change this in order to use more GPUs

./$PROGRAM 1000 2000

