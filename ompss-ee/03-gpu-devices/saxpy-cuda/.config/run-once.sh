PROGRAM=saxpy-p

export NX_GPUMAXMEM=90
export NX_GPUS=2 #change this in order to use more GPUs

./$PROGRAM

