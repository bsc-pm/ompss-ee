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

#ifdef __cplusplus
extern "C"
{
#endif

#pragma omp target device(cuda) copy_deps ndrange(1,nr,128)
#pragma omp task in([NA] a, [NH] h) out([NE] E) 
__global__ void cstructfac(int na, int nr, int nc, float f2, int NA,
                           TYPE_A*a, int NH, TYPE_H* h, int NE, TYPE_E*E);

#ifdef __cplusplus
}
#endif
