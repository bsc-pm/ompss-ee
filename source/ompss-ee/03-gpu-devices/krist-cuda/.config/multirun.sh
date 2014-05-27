PROGRAM=krist-p

export NX_GPUMAXMEM=150000000

export IFS=";"

GPUS="1"
ATOMS="1000;2000;3000;"
REFLECTIONS="2000"

for atoms in $ATOMS; do
  for reflections in $REFLECTIONS; do
    for NX_GPUS in $GPUS; do
      ./$PROGRAM $atoms $reflections
    done
  done
done

