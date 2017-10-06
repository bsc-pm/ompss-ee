PILS (Parallel ImbaLance Simulator)
-----------------------------------

.. highlight:: none

PILS is an MPI+OpenMP/OmpSs synthetic benchmark that measures the execution time
of imbalanced MPI ranks.

Usage::

    ./mpi_ompss_pils <loads-file> <parallel-grain> <loops> <task_size>
        loads-file:     file with load balance (number of tasks per iteration) per process, [100, 250] if /dev/null
        parallel-grain: parallelism grain, factor between 0..1 to apply sub-blocking techniques
        loops:          number of execution loops
        task_size:      factor to increase task size

**Goals of this exercise**

 * Run the instrumented version of PILS and generate a Paraver trace.

    * Analyse the load imbalance between MPI ranks.

 * Enable DLB and compare both executions.

    * Observe the dynamic thread creation when other processes suffer load imbalance.
    * Analyse the load imbalance of the new execution. Does it improve?

 * Enable DLB MPI interception and trace again. Analyse the new trace.
 * Run the multirun.sh script and compare the execution performance with and without DLB.
 * Modify the inputs of PILS to reduce load imbalance and see when DLB stops improving performance.

