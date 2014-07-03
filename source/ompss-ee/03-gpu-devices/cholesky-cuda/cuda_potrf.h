#ifndef _CUDA_POTRF_H_
#define _CUDA_POTRF_H_

#include <cublas_v2.h>

#ifdef __cplusplus
extern "C"{
#endif

int
cuda_dpotrf(cublasHandle_t handle, char uplo, int n, 
             double *dA, int ldda, int *info);

int
cuda_spotrf(cublasHandle_t handle, char uplo, int n, 
             float *dA, int ldda, int *info);

#ifdef __cplusplus
}
#endif

#endif // _CUDA_POTRF_H_