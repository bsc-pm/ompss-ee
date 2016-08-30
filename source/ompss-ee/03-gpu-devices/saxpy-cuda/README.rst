Saxpy kernel
------------

.. highlight:: c

In this exercise we will work with the Saxpy kernel. This algorithm sums one vector with another
vector multiplied by a constant.

The sources are not complete, but the standard structure for OmpSs CUDA/Kernel is complete:
 * There is a kernel in the files (kernel.cl/kernel.cu) in which the kernel code (or codes)
   is defined.
 * There is a C-file in which the host-program code is defined.
 * There is a kernel header file which declares the kernel as a task, this header must be
   included in the C-file and can also be included in the kernel file (not strictly needed).

Kernel header file (kernel.h) have::

  #pragma omp target device(cuda) copy_deps ndrange( /*???*/ )
  #pragma omp task in([n]x) inout([n]y)
  __global__ void saxpy(int n, float a,float* x, float* y);

As you can see, we have two vectors (x and y) of size n and a constant a. They specify which data
needs to be copied to our runtime. In order to get this program working, we only need to specify
the ``ndrange`` clause, which has three members:
 * First one is the number of dimensions on the kernel (1 in this case).
 * The second one is the total number of kernel threads to be launched (as one kernel thread
   usually calculates a single index of data, this is usually the number of elements, of the
   vectors in this case).
 * The third one is the group size (number of threads per block), in this kind of kernels which
   do not use shared memory between groups, any number from 16 to 128 will work correctly (optimal
   number depends on hardware, kernel code…).

When the ``ndrange`` clause is correct. We can proceed to compile the source code, using the command
'make'. After it (if there are no compilation/link errors), we can execute it using one of the
running scripts.

