#pragma omp target device(opencl) copy_deps ndrange( 1,n,128 )
#pragma omp task in([n]x) inout([n]y)
__kernel void saxpy(int n, float a,
       __global float* x, __global float* y);