Array Sum Benchmark (Fortran version)
----------------

.. highlight:: fortran

This benchmark computes the sum of two arrays and stores the result in an other array.

.. note::
  You can dowload this code visiting the url http://pm.bsc.es *OmpSs Examples and Exercises*'s
  (code) link. The Array Sum benchmark is included inside the  *01-examples*'s directory.

In this case we annotate the algorithm using the Fortran syntax. The benchmark compute
a set of array sums. The first inner loop initializes one array, that will be computed
in the second inner loop. Dependences warrant proper execution and synchronization
between initialization and compute results::

   DO K=1,1000
      IF(MOD(K,100)==0) WRITE(0,*) 'K=',K
      ! INITIALIZE THE ARRAYS
      DO I=1, N, BS
         !$OMP TASK OUT(VEC1(I:I+BS-1), VEC2(I:I+BS-1), RESULTS(I:I+BS-1))&
         !$OMP PRIVATE(J) FIRSTPRIVATE(I, BS) LABEL(INIT_TASK)
         DO J = I, I+BS-1
            VEC1(J) = J
            VEC2(J) = N + 1 - J
            RESULTS(J) = -1
         END DO
         !$OMP END TASK
      ENDDO
      ! RESULTS = VEC1 + VEC2
      DO I=1, N, BS
         !$OMP TASK OUT(VEC1(I:I+BS-1), VEC2(I:I+BS-1)) IN(RESULTS(I:I+BS-1))&
         !$OMP PRIVATE(J) FIRSTPRIVATE(I, BS) LABEL(ARRAY_SUM_TASK)
         DO J = I, I+BS-1
            RESULTS(J) = VEC1(J) + VEC2(J)
         END DO
         !$OMP END TASK
      ENDDO
   ENDDO ! K
   !$OMP TASKWAIT
