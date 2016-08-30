Heat diffusion (Jacobi solver)
---------------

.. highlight:: c

This codes performs a ... hybrid MPI/OmpSs implementation.

.. note::
  You can dowload this code visiting the url http://pm.bsc.es *OmpSs Examples and Exercises*'s
  (code) link. This version of matrix multiply kernel is included inside the  *04-mpi+ompss*'s directory.

.. note::
  You need to specify the number of MPI tasks per node. In Marenostrum you can do this
  by adding <<#BSUB -R "span[ptile=1]">> to your job script.
