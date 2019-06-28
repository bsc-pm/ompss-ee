PROGRAM=nbody-p

export NX_SMP_WORKERS=1

for gpus in 1 2; do
  export NX_GPUS=$gpus
  NX_ARGS="--cache-policy writethrough --gpu-max-memory 90" ./$PROGRAM nbody_input-16384.in
done

