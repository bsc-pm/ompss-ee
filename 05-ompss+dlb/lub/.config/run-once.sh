PROGRAM=LUB-i

# Uncomment to instrument
# export INST=./trace.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask --lend-mode=BLOCK"
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi_instr.so

export NX_ARGS+=" --force-tie-master --warmup-threads"

mpirun $INST ./$PROGRAM 2000 100

