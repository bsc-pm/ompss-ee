Cholesky kernel
---------------

.. highlight:: c

This example shows the Cholesky kernel. This algorithm is a decomposition of a Hermitian,
positive-definite matrix into the product of a lower triangular matrix and its conjugate
transpose.

.. note::
  You can dowload this code visiting the url http://pm.bsc.es *OmpSs Examples and Exercises*'s
  (code) link. The Cholesky kernel is included inside the  *01-examples*'s directory.

The kernel uses four different linear algorithms: potrf, trsm, gemm and syrk.
Following code shows the basic pattern for a Cholesky factorisation::

   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization
      omp_potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++)
         omp_trsm (Ah[k][k], Ah[k][i], ts, ts);

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++)
            omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);

         omp_syrk (Ah[k][i], Ah[i][i], ts, ts);
      }

   }

In this case we parallelize the code by annotating the kernel functions.
So each call in the previous loop becomes the instantiation of a task.
The following code shows how we have parallelized Cholesky::

   #pragma omp task inout([ts][ts]A)
   void omp_potrf(double * const A, int ts, int ld)
   {
      ...
   }

   #pragma omp task in([ts][ts]A) inout([ts][ts]B)
   void omp_trsm(double *A, double *B, int ts, int ld)
   {
      ...
   }

   #pragma omp task in([ts][ts]A) inout([ts][ts]B)
   void omp_syrk(double *A, double *B, int ts, int ld)
   {
      ...
   }

   #pragma omp task in([ts][ts]A, [ts][ts]B) inout([ts][ts]C)
   void omp_gemm(double *A, double *B, double *C, int ts, int ld)
   {
      ...
   }

Note that for each of the dependences we also specify which is the matrix (block) size.
Although this is not needed, due there is no overlapping among the different blocks,
it will allow the runtime to compute dependences using the region mechanism.
