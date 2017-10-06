Lulesh
------

.. highlight:: none

Lulesh is a benchmark from LLNL, it represents a typical hydrocode like ALE3D.

Usage::

    ./lulesh2.0 -i <iterations> -b <balance> -s <size>


**Goals of this exercise**

 * Run the instrumented version of Lulesh and analyse the Paraver trace.
 * Enable DLB options, MPI interception included. Run and analyse the Paraver trace.
 * Run the multirun.sh script and compare the execution performance with and without DLB.
