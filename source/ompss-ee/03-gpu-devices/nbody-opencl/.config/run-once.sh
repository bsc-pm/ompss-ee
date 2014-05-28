PROGRAM=nbody-p

export NX_THREADS=1
export NX_OPENCL_MAX_DEVICES=2 #max number of opencl devices (GPUs in this case) to use

NX_ARGS="--cache-policy wt --gpu-max-memory 1000000000" ./$PROGRAM nbody_input-16384.in
