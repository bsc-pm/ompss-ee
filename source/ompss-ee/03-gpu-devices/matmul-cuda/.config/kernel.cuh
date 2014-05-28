
// Thread block size
#define BLOCK_SIZE 16


#ifdef DP
#define REAL double
#else
#define REAL float
#endif

#ifdef __cplusplus
extern "C"
{
#endif

//Kernel declaration as a task should be here
//Remember, we want to multiply two matrices, (A*B=C) where all of them have size NB*NB
#pragma omp target device(cuda) ndrange(2,NB,NB,16,16) copy_deps
#pragma omp task inout([NB*NB]C) in([NB*NB]A,[NB*NB]B)
__global__ void Muld(REAL* A, REAL* B, int wA, int wB, REAL* C,int NB)

#ifdef __cplusplus
}
#endif
