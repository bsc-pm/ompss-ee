PROGRAM=matmul-p

# Run with 6 threads per MPI process in the same node
export NX_SMP_WORKERS=6

# Uncomment to instrument
#export INST=./graph.sh
#export INST=./trace.sh

${MPIRUN_COMMAND} $INST ./$PROGRAM

# Generate the trace if needed
if [[ "$INST" == *"trace"* ]]; then
	mpi2prv -f TRACE.mpits -o myTrace.prv
fi
