#include "kernel_header.h"

__kernel void saxpy(int n, float a,
          __global float* x, __global float* y) {
    int i = get_global_id(0);
    if(i < n)
       y[i] = a * x[i] + y[i];
}