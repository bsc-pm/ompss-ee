PROGRAM=heat-p

export NX_ARGS="--threads 2"

echo "##################################################"
echo "NX_ARGS is $NX_ARGS"
echo "LD_PRELOAD is $LD_PRELOAD"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "##################################################"

mpirun ./$PROGRAM test.dat test.ppm

