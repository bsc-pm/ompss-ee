
#define DIM2_H 4
#define DIM2_A 4
#define DIM2_E 2

#if DIM2_H == 4
#define TYPE_H float4
#endif
#if DIM2_H == 3
#define TYPE_H float3
#endif
#if DIM2_A == 4
#define TYPE_A float4
#endif
#if DIM2_A == 3
#define TYPE_A float3
#endif
#if DIM2_E == 4
#define TYPE_E float4
#endif
#if DIM2_E == 3
#define TYPE_E float3
#endif
#if DIM2_E == 2
#define TYPE_E float2
#endif

#ifndef __OPENCL_VERSION__
#pragma omp target device(opencl) copy_deps ndrange(/*???*/)
#pragma omp task /* in and outs? */
__kernel void cstructfac(int na, int nr, int nc, float f2,
                          int NA, __global float* a, int NH , __global float* h, int NE, __global float* E);
#endif

