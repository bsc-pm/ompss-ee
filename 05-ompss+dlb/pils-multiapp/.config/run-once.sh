PROGRAM=ompss_pils-i

# Uncomment to instrument
# export INST=./trace-multiapp.sh

# Uncomment to enable DLB
# export NX_ARGS+=" --thread-manager=dlb"
# export DLB_ARGS+=" --policy=auto_LeWI_mask"

export NX_ARGS+=" --warmup-threads"

export TRACEID=TRACE1
taskset -c 0-7 $INST ./$PROGRAM input1 1 100 500 &

export TRACEID=TRACE2
taskset -c 8-15 $INST ./$PROGRAM input2 1 100 50 &

wait

if [[ -n "$INST" ]] ; then
    mpi2prv -f TRACE1.mpits -- -f TRACE2.mpits -o myTrace.prv
fi
