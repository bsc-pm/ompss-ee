#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "krist.h"


__global__ void cstructfac(int na, int nr, int nc, float f2, int NA,
                           TYPE_A*a, int NH, TYPE_H* h, int NE, TYPE_E*E)
{
	__shared__ TYPE_A ashared[(16384-2048)/sizeof(TYPE_A)];
    int a_start;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nr) E[i].x = E[i].y = 0.0f;

    for (a_start = 0; a_start < na; a_start += nc) {
        int a_end = min(a_start + nc, na);
        int k = threadIdx.x;
        while (k < a_end - a_start) {
            ashared[k] = a[k + a_start];
            k += blockDim.x;
        }

        __syncthreads();

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
                sincosf(arg, &B1, &A1);
                A += aj.x*A1;
                B += aj.x*B1;
            }
            E[i].x += A*f2;
            E[i].y += B*f2;
        }
        __syncthreads();
    }
}
