PROGRAM=stream-barr-p

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"

for thread in $THREADS; do
  NX_GPUS=0 NX_SMP_WORKERS=$thread ./$PROGRAM
done
