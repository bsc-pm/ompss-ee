PROGRAM=mpi_ompss_pils-p

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask"

export NX_ARGS+=" --force-tie-master --warmup-threads"

for i in $(seq 1 3) ; do
    mpirun ./$PROGRAM /dev/null 1 10 500 | grep 'Application time'
done
