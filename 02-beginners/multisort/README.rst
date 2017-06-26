Multisort application
---------------------

.. highlight:: c

Multisort application, sorts an array using a divide and conquer strategy. The vector is split
into 4 chunks, and each chunk is recursively sorted (as it is recursive, it may be even split
into other 4 smaller chunks), and then the result is merged. When the vector size is smaller
than a configurable threshold (MIN_SORT_SIZE) the algorithm switches to a sequential sort
version::

  if (n >= MIN_SORT_SIZE*4L) {

    // Recursive decomposition
    multisort(n/4L, &data[0], &tmp[0]);
    multisort(n/4L, &data[n/4L], &tmp[n/4L]);
    multisort(n/4L, &data[n/2L], &tmp[n/2L]);
    multisort(n/4L, &data[3L*n/4L], &tmp[3L*n/4L]);

    // Recursive merge: quarters to halves
    merge(n/4L, &data[0], &data[n/4L], &tmp[0], 0, n/2L);
    merge(n/4L, &data[n/2L], &data[3L*n/4L], &tmp[n/2L], 0, n/2L);

    // Recursive merge: halves to whole
    merge(n/2L, &tmp[0], &tmp[n/2L], &data[0], 0, n);

  } else {
    // Base case: using simpler algorithm
    basicsort(n, data);
  }

.. highlight:: none

As the code is already annotated with some task directives, try to compile and run the program.
Is it verifying? Why do you think it is failing? Running an unmodified version of this code may
also raise a ``Segmentation Fault``. Investigate which is the cause of that problem. Although it
is not needed, you can also try to debug program execution using gdb debugger (with the OmpSs
debug version)::

  $NX_THREADS=4 gdb --args ./multisort-d 4096 64 128

**Goals of this exercise**

 * Solve the existant bug, program is not properly annotated.
 * Think how the tasks must be synchronized and annotate the source file.
 * Check different parallelization approaches: taskwait/dependences.
 * Check scalability (for the different versions), use other runtime options (schedulers,... )
 * Get a task dependency graph (different domains) and/or paraver traces

