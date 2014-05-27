#include "krist_auxiliar_header.h"


#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void cstructfac(int na, int nr, int nc, float f2, int NA,
                           __global TYPE_A* a,int NH, __global TYPE_H* h,int NE,__global TYPE_E* E)
{
	__local TYPE_A ashared[(16384-2048)/(sizeof(TYPE_A))];
    int a_start;

    int i = get_global_id(0);
    if (i < nr) E[i].x = E[i].y = 0.0f;

    for (a_start = 0; a_start < na; a_start += nc) {
        int a_end = min(a_start + nc, na);
        int k = get_local_id(0);
        while (k < a_end - a_start) {
            ashared[k] = a[k + a_start];
            k += get_local_size(0);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < nr) {
            int j;
            float A,B;
            const float twopi = 6.28318584f;

            TYPE_H hi  = h[i];
            A          = 0.0f;
            B          = 0.0f;

            int jmax = a_end - a_start;
            for (j=0; j < jmax; j++) {
                float A1,B1;
                float4 aj = ashared[j];
                float arg = twopi*(hi.x*aj.y +
                                   hi.y*aj.z +
                                   hi.z*aj.w);
                B1=sincos(arg, &A1);
                A += aj.x*A1;
                B += aj.x*B1;
            }
            E[i].x += A*f2;
            E[i].y += B*f2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
