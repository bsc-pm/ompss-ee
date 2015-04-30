PROGRAM=matmul-i

export NX_ARGS="--threads 2"

echo "NX_ARGS is $NX_ARGS"
echo "LD_PRELOAD is $LD_PRELOAD"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "##################################################"

# Uncomment to enable tracing
#export INST=./trace.sh

mpirun --cpus-per-proc 2 $INST ./$PROGRAM

