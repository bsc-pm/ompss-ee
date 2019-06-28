PROGRAM=saxpy-p

export NX_GPUMAXMEM=90

for gpus in 1 2; do
  export NX_GPUS=$gpus
  ./$PROGRAM
done

