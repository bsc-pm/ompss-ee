Cholesky kernel
---------------

This kernel is just like the SMP version found in the examples, but implemented
in CUDA. It uses CUBLAS kernels for the ``syrk``, ``trsm`` and ``gemm``
kernels, and a CUDA implementation for the potrf kernel (declared in a
different file).

Your assignment is to annotate all CUDA tasks in the source code under the
section "TASKS FOR CHOLESKY".

