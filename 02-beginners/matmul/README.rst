Matrix Multiplication
---------------------

.. highlight:: c

This example performs the multiplication of two matrices (A and B) into a third one (C). Since
the code is not optimized, not very good performance results are expected. Think about how to
parallelize (using OmpSs) the following code found in compute() function::

  for (i = 0; i < DIM; i++)
    for (j = 0; j < DIM; j++)
      for (k = 0; k < DIM; k++)
        matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);

This time you are on your own: you have to identify what code must be a task. There are a few
hints and that you may consider before do the exercise:

 * Have a look at the compute function. It is the one that the main procedure calls to perform
   the multiplication. As you can see, this algorithm operates on blocks (to increase memory
   locality and to parallelize operations on those blocks).
 * Now go to the matmul function. As you can see, this function performs the multiplication on
   a block level.
 * When creating tasks do not forget to ensure that all of them have finished before returning
   the result of the matrix multiplication (would it be necessary any synchronization directive
   to guarantee that result has been already computed?).

**Goals of this exercise**

 * Look for candidates to become a task and taskify them
 * Include synchroniztion directives when required
 * Check scalability (for different versions), use different runtime options (schedulers,... )
 * Get a task dependency graph and/or paraver traces




