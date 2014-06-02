PROGRAM=matmul-p

export NX_ARGS="--pes 2 --disable-cuda=yes"

echo "NX_ARGS is $NX_ARGS"
echo "LD_PRELOAD is $LD_PRELOAD"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "##################################################"

srun ./$PROGRAM

