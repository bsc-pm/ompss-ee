PROGRAM=multisort-p

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"
VSIZE="65536"
SEQ_SORT="256"
SEQ_MERGE="512"

for size in $VSIZE; do
  for seq_sort in $SEQ_SORT; do
    for seq_merge in $SEQ_MERGE; do
      for thread in $THREADS; do
        NX_GPUS=0 NX_SMP_WORKERS=$thread ./$PROGRAM $size $seq_sort $seq_merge
      done
    done
  done
done
