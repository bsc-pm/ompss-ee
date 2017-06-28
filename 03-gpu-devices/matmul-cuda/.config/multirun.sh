PROGRAM=matmul-p

export IFS=";"

export NX_GPUMAXMEM=90

GPUS="01;02"
SIZES="8192"

for size in $SIZES; do
  # Creating the input file
  touch test.in
  echo "$size $size $size 3" > test.in
  for gpu in $GPUS; do
    # Executing the application
    NX_GPUS=$gpu NX_SMP_WORKERS=1 ./$PROGRAM
  done
done

