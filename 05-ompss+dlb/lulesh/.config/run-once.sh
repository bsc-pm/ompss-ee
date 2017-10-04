PROGRAM=lulesh2.0-i

# Uncomment to instrument
# export INST=./trace.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask --lend-mode=BLOCK"
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi_instr.so
# export I_MPI_WAIT_MODE=1

export NX_ARGS+=" --force-tie-master --warmup-threads"
mpirun -n 27 $INST ./$PROGRAM -i 15 -b 8 -s 100
