PROGRAM=mpi_ompss_pils-i

# Uncomment to instrument
# export INST=./trace.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export LB_POLICY="auto_LeWI_mask"

# Uncomment to enable DLB MPI interception
# export OMPSSEE_LD_PRELOAD=$DLB_HOME/lib/libdlb_mpi_instr.so

mpirun $INST ./$PROGRAM /dev/null 1 10 500
