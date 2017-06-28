PROGRAM=array_sum-p

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"
NSIZES="8388608"
BSIZES="32768"

for N in $NSIZES; do
    for BS in $BSIZES; do
        for thread in $THREADS; do
            NX_SMP_WORKERS=$thread ./$PROGRAM $N $BS
        done
    done
done
