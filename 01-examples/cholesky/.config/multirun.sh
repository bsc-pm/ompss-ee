
PROGRAM=cholesky-p

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"
MSIZES="2048"
BSIZES="256"

for MS in $MSIZES; do
  for BS in $BSIZES; do
    for thread in $THREADS; do
      NX_SMP_WORKERS=$thread ./$PROGRAM $MS $BS 0
    done
  done
done
