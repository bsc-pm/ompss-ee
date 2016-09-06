#ifndef __CUDACC__

void matmul ( int m, int n, double (*A)[m], double (*B)[n], double (*C)[n] );

#endif // __CUDACC__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__

void dgemm_cublas (int BS, double *A, double *B, double *C);

#else

extern int BS;

//#pragma omp target device (cuda) copy_deps
//#pragma omp task  input([n][m]A, [m][n]B)  inout([m][n]C)
//void dgemm_cublas (int m, int n, double *A, double *B, double *C);


#endif

#ifdef __cplusplus
}
#endif
