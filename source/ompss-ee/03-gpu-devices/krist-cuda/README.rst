Krist kernel
------------

.. highlight:: c

Krist kernel is used on crystallography to find the exact shape of a molecule using Rntgen
diffraction on single crystals or powders. We’ll execute the same kernel many times.

The sources are not complete, but the standard structure for OmpSs CUDA/Kernel is complete:

 * There is a kernel in the files (kernel.cl/kernel.cu) in which the kernel code (or codes)
   is defined.
 * There is a C-file in which the host-program code is defined. 
 * There is a kernel header file (krist.h) which declares the kernel as a task, this header
   must be included in the C-file and can also be included in the kernel file (not strictly
   needed).

Krist header file (krist.h) have::

  #pragma omp target device(cuda) copy_deps //ndrange?
  #pragma omp task //in and outs?
  __global__ void cstructfac(int na, int number_of_elements, int nc, float f2, int NA,
                             TYPE_A* a, int NH, TYPE_H* h, int NE, TYPE_E* output_array);

As you can see, now we need to specify the ndrange clause (same procedure than previous exercise)
and the inputs and outputs. As we have done before for SMP (hint: Look at the source code of the
kernel in order to know which arrays are read and which ones are written). The total number of
elements which we’ll process (not easy to guess by reading the kernel) is 'number_of_elements'.

Remind: ND-range clause has three members:

 * First one is the number of dimensions on the kernel (1 in this case).
 * The second one is the total number of kernel threads to be launched (as one kernel threads
   usually calculates a single index of data, this is usually the number of elements, of the
   vectors in this case).
 * The third one is the group size (number of threads per block), in this kind of kernels which
   do not use shared memory between groups, any number from 16 to 128 will work correctly (optimal
   number depends on hardware, kernel code...

Once the ndrange clause is correct and the input/outputs are correctly defined. We can proceed to
compile the source code, using the command 'make'. After it (if there are no errors), we can
execute it using one of the provided runnin scripts. Check if all environment variables are set to
the proper values.

