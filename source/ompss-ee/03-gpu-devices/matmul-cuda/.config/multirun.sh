PROGRAM=matmul-p

export IFS=";"

export NX_GPUMAXMEM=150000000

GPUS="01;02"
SIZES="8192"

for size in $SIZES; do
  # Creating the input file
  touch test.in
  echo "$size $size $size 3" > test.in
  for gpu in $GPUS; do
    # Executing the application
    NX_GPUS=$gpu NX_THREADS=1 ./$PROGRAM
  done
done

