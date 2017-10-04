PROGRAM=lulesh2.0-i

# Uncomment to instrument
# export INST=./trace.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export LB_POLICY="auto_LeWI_mask"
# export LB_LEND_MODE="BLOCK"
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi_instr.so
# export I_MPI_WAIT_MODE=1

export NX_ARGS+=" --force-tie-master --warmup-threads"
mpirun -n 27 $INST ./$PROGRAM -i 15 -b 8 -s 100
