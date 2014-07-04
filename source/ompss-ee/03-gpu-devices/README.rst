GPU Device Exercises
====================

Almost all the programs in this section is available both in OpenCL and CUDA. From the point of
view of an OmpSs programmer, the only difference between them is the language in which the kernel
is written.

As OmpSs abstracts the user from doing the work in the host part of the code. Both OpenCL and CUDA
have the same syntax. You can do any of the two versions, as they are basically the same, when you
got one of them working, same steps can be done in the other version. 

.. toctree::
   :maxdepth: 2
   :numbered:

   saxpy-cuda/README.rst
   krist-cuda/README.rst
   matmul-cuda/README.rst
   nbody-cuda/README.rst
   cholesky-cuda/README.rst
