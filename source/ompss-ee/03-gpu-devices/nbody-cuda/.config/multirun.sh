PROGRAM=nbody-p

export NX_THREADS=1
export NX_GPUS=2 #change this in order to use more GPUs

NX_ARGS="--cache-policy wt --gpu-max-memory 1000000000" ./$PROGRAM nbody_input-16384.in

