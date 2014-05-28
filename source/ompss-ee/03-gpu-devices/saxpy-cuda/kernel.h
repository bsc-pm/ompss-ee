
#ifdef __cplusplus
extern "C"
{
#endif

#pragma omp target device(cuda) copy_deps ndrange(/*???*/)
#pragma omp task in([n]x) inout([n]y)
__global__ void saxpy(int n, float a,float* x, float* y);

#ifdef __cplusplus
}
#endif
