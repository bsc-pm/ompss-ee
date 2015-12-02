export IFS=";"

THREADS="01;02;04;08;12"

#serial
PROGRAM=multisort-serial

./$PROGRAM 32768 512 512

VERSION=multisort-leaf
#VERSION=multisort-tree
PROGRAM=$VERSION-p

for thread in $THREADS; do
  OMP_NUM_THREADS=$thread ./$PROGRAM 32768 512 512
  
  cmp $VERSION.out multisort-serial.out
  if [ $? == 0 ]; then
    echo success
  else
    echo differ!!
  fi
done
