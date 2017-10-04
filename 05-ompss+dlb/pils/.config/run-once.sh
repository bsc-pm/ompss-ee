PROGRAM=mpi_ompss_pils-i

# Uncomment to instrument
# export INST=./trace.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask"

# Uncomment to enable DLB MPI interception
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi_instr.so

export NX_ARGS+=" --force-tie-master"
mpirun $INST ./$PROGRAM /dev/null 1 10 500
