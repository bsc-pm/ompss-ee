#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "omp.h"

#include "cholesky.h"

#pragma omp task inout([ts][ts]A)
void omp_potrf(double * const A, int ts, int ld)
{
   static int INFO;
   static const char L = 'L';
   dpotrf_(&L, &ts, A, &ld, &INFO);
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_trsm(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
   static double DONE = 1.0;
   dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_syrk(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}

#pragma omp task in([ts][ts]A, [ts][ts]B) inout([ts][ts]C)
void omp_gemm(double *A, double *B, double *C, int ts, int ld)
{
   static const char TR = 'T', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}

void cholesky_blocked(const int ts, const int nt, double* Ah[nt][nt])
{
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      omp_potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         omp_trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         omp_syrk (Ah[k][i], Ah[i][i], ts, ts);
      }

   }
#pragma omp taskwait
}

int main(int argc, char* argv[])
{
   char *result[3] = {"n/a","sucessful","UNSUCCESSFUL"};
   const double eps = BLAS_dfpinfo( blas_eps );

   if ( argc < 4) {
      printf( "cholesky matrix_size block_size check\n" );
      exit( -1 );
   }
   const int  n = atoi(argv[1]); // matrix size
   const int ts = atoi(argv[2]); // tile size
   int check    = atoi(argv[3]); // check result?

   // Allocate matrix
   double * const matrix = (double *) malloc(n * n * sizeof(double));
   assert(matrix != NULL);

   // Init matrix
   initialize_matrix(n, ts, matrix);

   // Allocate matrix
   double * const original_matrix = (double *) malloc(n * n * sizeof(double));
   assert(original_matrix != NULL);

   const int nt = n / ts;

   // Allocate blocked matrix
   double *Ah[nt][nt];

   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         Ah[i][j] = malloc(ts * ts * sizeof(double));
         assert(Ah[i][j] != NULL);
      }
   }

   for (int i = 0; i < n * n; i++ ) {
      original_matrix[i] = matrix[i];
   }

#ifdef VERBOSE
   printf ("Executing ...\n");
#endif

   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);

   const float t1 = get_time();
   cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);

   const float t2 = get_time() - t1;
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);

   if ( check ) {
      const char uplo = 'L';
      if ( check_factorization( n, original_matrix, matrix, n, uplo, eps) ) check++;
   }

   free(original_matrix);

   float time = t2;
   float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

   // Print results
#ifdef VERBOSE
   printf( "============ CHOLESKY RESULTS ============\n" );
   printf( "  matrix size:          %dx%d\n", n, n);
   printf( "  block size:           %dx%d\n", ts, ts);
   printf( "  number of threads:    %d\n", omp_get_num_threads());
   printf( "  time (s):             %f\n", time);
   printf( "  performance (gflops): %f\n", gflops);
   printf( "==========================================\n" );
#else
   printf( "test:%s-%d-%d:threads:%2d:result:%s:gflops:%f\n", argv[0], n, ts, omp_get_num_threads(), result[check], gflops );
#endif

   // Free blocked matrix
   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         assert(Ah[i][j] != NULL);
         free(Ah[i][j]);
      }
   }

   // Free matrix
   free(matrix);

   return 0;
}

