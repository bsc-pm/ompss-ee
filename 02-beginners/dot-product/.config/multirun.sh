PROGRAM=dot-product-p

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"
MSIZE="8192"
BSIZE="128"

for MS in $MSIZE; do
  for BS in $BSIZE; do
    for thread in $THREADS; do
      NX_GPUS=0 NX_SMP_WORKERS=$thread ./$PROGRAM $MS $BS
    done
  done
done
