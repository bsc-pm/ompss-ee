PROGRAM=LUB-p

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask --lend-mode=BLOCK"
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi.so

export NX_ARGS+=" --force-tie-master --warmup-threads"

for i in $(seq 1 3) ; do
    mpirun env LD_PRELOAD=$OMPSSEE_LD_PRELOAD ./$PROGRAM 8000 100 | grep 'time to compute'
done
