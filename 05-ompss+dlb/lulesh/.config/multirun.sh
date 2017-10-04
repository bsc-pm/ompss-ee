PROGRAM=lulesh2.0-p

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask --lend-mode=BLOCK"
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi.so
# export I_MPI_WAIT_MODE=1

export NX_ARGS+=" --force-tie-master --warmup-threads"

for i in $(seq 1 3) ; do
    mpirun -n 27 env LD_PRELOAD=$OMPSSEE_LD_PRELOAD ./$PROGRAM -i 15 -b 8 -s 100 \
        | tac | grep -m 1 'Elapsed time'
done
