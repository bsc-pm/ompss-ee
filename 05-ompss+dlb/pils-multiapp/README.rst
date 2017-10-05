PILS - multiapp example
-----------------------

.. highlight:: c

This example demonstrates the capabilities of DLB sharing resources with two different
unrelated applications. The run-once.sh script executes two instances of PILS without
MPI support, each one in a different set of CPUs. DLB is able to automatically lend
resources from one to another.

**Goals of this exercise**

 * Run the script run-once.sh with tracing and DLB enabled, and observe how two
   unrelated applications share resources.
