PROGRAM=heat-p

export IFS=";"

THREADS="01;02;04;08;12"
INPUT=test512.dat

for thread in $THREADS; do
  OMP_NUM_THREADS=$thread ./$PROGRAM $INPUT test512-$thread.ppm
done
