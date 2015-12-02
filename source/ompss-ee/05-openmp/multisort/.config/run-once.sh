#PROGRAM=multisort-leaf-p
#PROGRAM=multisort-tree-p
# Uncomment to instrument
PROGRAM=multisort-leaf-i
#PROGRAM=multisort-tree-i

# Run with 2 threads
export OMP_NUM_THREADS=2

# Uncomment to instrument
INST=./trace.sh

$INST ./$PROGRAM 32768 512 512
