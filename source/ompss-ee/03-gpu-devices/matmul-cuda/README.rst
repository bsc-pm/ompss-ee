Matrix Multiply
---------------

.. highlight:: c

In this exercise we will work with the Matmul kernel. This algorithm is used to multiply two
2D-matrices and store the result in a third one. Every matrix has the same size. 

This is a blocked-matmul multiplications, this means that we launch many kernels and each one
of this kernels will multiply a part of the matrix. This way we can increase parallelism, by
having many kernels which may use as many GPUs as possible.

Sources are not complete, but the standard file structure for OmpSs CUDA/Kernel is complete:
 * There is a kernel in the files (kernel.cl/kernel.cu) in which the kernel code (or codes)
   is defined.  
 * There is a C-file in which the host-program code is defined. 
 * There is a kernel header file which declares the kernel as a task, this header must be
   included in the C-file and can also be included in the kernel file (not strictly needed).
  
Matrix multiply header file (matmul.h) have::

  //Kernel declaration as a task should be here
  //Remember, we want to multiply two matrices, (A*B=C) where all of them have size NB*NB

In this header, there is no kernel declared as a task, you have to search into the kernel.cu/cl
file in order to see which kernel you need to declare, declare the kernel as a task, by placing
its declaration and the pragmas over it.

.. note::
In this case as we are multiplying a two-dimension matrix, so the best approach is to use a
two-dimension ndrange.

In order to get this program working, we only need to specify the NDRange clause, which has
five members:
 * First one is the number of dimensions on the kernel (2 in this case). 
 * The second and third ones are the total number of kernel threads to be launched (as one
   kernel threads usually calculates a single index of data, this is usually the number of
   elements, of the vectors in this case) per dimension.
 * The fourth and fifth ones are the group size (number of threads per block), in this kind
   of kernels which do not use shared memory between groups, any number from 16 to 32 (per
   dimension) should work correctly (depending on the underlying Hardware).

Once the ndrange clause is correct and the input/outputs are correctly defined. We can proceed
to compile the source code, using the command 'make'. After it (if there are no errors), we can
execute it using one of the running scripts.

