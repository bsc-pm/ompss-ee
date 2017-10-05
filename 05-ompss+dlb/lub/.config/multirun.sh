PROGRAM=LUB-p

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask"

export NX_ARGS+=" --force-tie-master --warmup-threads"

for i in $(seq 1 3) ; do
    mpirun $INST ./$PROGRAM 8000 100 | grep 'time to compute'
done
