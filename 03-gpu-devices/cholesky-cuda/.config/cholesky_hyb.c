#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <errno.h>

#define CUDA_VERSION 5000

#ifndef CUDA_VERSION
#error Cuda version not defined
#endif

int NUM_NODES;
//#define nanos_current_socket(x)


// Configuration options
#ifdef CHOL_BASE
#define CHOL_VERSION_DEFINED 0
#define CONVERT_TASK 0
#define CONVERT_REQUEST 0
#define POTRF_SMP 0
#define POTRF_NESTED 0
#define CUDA_POTRF 1
#endif

#ifdef CHOL_CONVERT_UNDER_REQUEST
#define CHOL_VERSION_DEFINED 3
#define CONVERT_TASK 1
#define CONVERT_REQUEST 1
#define POTRF_SMP 0
#define POTRF_NESTED 0
#endif

// Configuration variables defined from compile line
#ifdef CHOL_COMPILE_CONF
#define CHOL_VERSION_DEFINED 5
#endif


#ifdef CUSTOM_CONFIG
  #define CHOL_VERSION_DEFINED 6
  
  #ifdef CT
    #define CONVERT_TASK 1
  #else
    #define CONVERT_TASK 0
  #endif
  
  #ifdef PS
    #define POTRF_SMP 1
  #else
    #define POTRF_SMP 0
  #endif
  
  #ifdef PN
    #define POTRF_NESTED 1
  #else
    #define POTRF_NESTED 0
  #endif
  
  #ifdef PRI
    #define USE_PRIORITY 1
  #else
    #define USE_PRIORITY 0
  #endif
  
  #ifdef PIN
    #define USE_PINNED 1
  #else
    #define USE_PINNED 0
  #endif

  #ifdef IMP
    #define USE_IMPLEMENTS 1
  #else
    #define USE_IMPLEMENTS 0
  #endif

  #define CONVERT_REQUEST 0

#endif

// Default
#ifndef CHOL_VERSION_DEFINED
#define CONVERT_TASK    1
#define CONVERT_REQUEST 0 // Not working...
#define POTRF_SMP       0
#define POTRF_NESTED    0
#define USE_PRIORITY    1
#define USE_PINNED      1
#define USE_IMPLEMENTS  0
#define CUDA_POTRF      1
#endif

#ifndef USE_IMPLEMENTS
#define USE_IMPLEMENTS 0
#endif


// Checking constraints
// If converting matrix under request, mark convert functions as tasks
#if CONVERT_REQUEST
 #if !CONVERT_TASK
  #undef CONVERT_TASK
  #define CONVERT_TASK 1
 #endif
#endif

// Include GPU kernel's library
#if CUDA_VERSION < 5000
# include <cublas.h>
#else
# include <cublas_v2.h>
#endif

#if CUDA_POTRF
#  include "cuda_potrf.h"
#endif

// Define macro's to make the code cleaner
#if POTRF_NESTED
 #define CALL_POTRF_TILE(x1, x2) smp_cholesky(x1, x2);
#else
 #define CALL_POTRF_TILE(x1, x2) potrf_tile(x1, x2);
#endif

#if CONVERT_REQUEST
 #define TRY_GATHER_BLOCK(x1, x2, x3, x4) \
    if (x4 == NULL) { \
        x4 = malloc_block(x2); \
        gather_block(x1, x2, x3, x4); \
    }

 #define CHECK_BLOCK_NOT_NULL(x1) if (A[i][j] != NULL)
#else
 #define TRY_GATHER_BLOCK(x1, x2, x3, x4) 
 #define CHECK_BLOCK_NOT_NULL(x1) 
#endif

#ifdef DOUBLE_PREC
#define REAL double
#define laset_		dlaset_
#define gemm_		dgemm_
#define lange_		dlange_
#define larnv_		dlarnv_
#define potrf_		dpotrf_
#define trsm_		dtrsm_
#define syrk_		dsyrk_
#define gpu_potrf       gpu_d_potrf
#define gpu_blas_gemm	gpu_blas_d_gemm
#define gpu_blas_trsm	gpu_blas_d_trsm
#define gpu_blas_syrk	gpu_blas_d_syrk
#define accepted_error	3.0e-15
#define PRINT_PRECISION printf("\tCholesky double precision\n");
#else
#define REAL float
#define laset_		slaset_
#define gemm_		sgemm_
#define lange_		slange_
#define larnv_		slarnv_
#define potrf_		spotrf_
#define trsm_		strsm_
#define syrk_		ssyrk_
#define gpu_potrf       gpu_s_potrf
#define gpu_blas_gemm	gpu_blas_s_gemm
#define gpu_blas_trsm	gpu_blas_s_trsm
#define gpu_blas_syrk	gpu_blas_s_syrk
#define accepted_error	1.0e-1
#define PRINT_PRECISION printf("\tCholesky single precision\n");
#endif


#define gpu_blas_d_gemm cublasDgemm
#define gpu_blas_d_trsm cublasDtrsm
#define gpu_blas_d_syrk cublasDsyrk
#define gpu_blas_s_gemm cublasSgemm
#define gpu_blas_s_trsm cublasStrsm
#define gpu_blas_s_syrk cublasSsyrk
#define gpu_s_potrf  cuda_spotrf
#define gpu_d_potrf  cuda_dpotrf

#if USE_PINNED
#define my_malloc nanos_malloc_pinned_cuda
#define my_free   nanos_free_pinned_cuda
#else
#define my_malloc malloc
#define my_free   free
#endif

#if USE_IMPLEMENTS
#undef POTRF_SMP
#define POTRF_SMP 0
#define POTRF_GPU 1
#define GEMM_SMP 1
#define GEMM_GPU 1
#define TRSM_SMP 0
#define TRSM_GPU 1
#define SYRK_SMP 0
#define SYRK_GPU 1
#else
#define GEMM_SMP 0
#define GEMM_GPU 1
#define TRSM_SMP 0
#define TRSM_GPU 1
#define SYRK_SMP 0
#define SYRK_GPU 1
#endif

#if !POTRF_SMP
#define POTRF_GPU 1
#define potrf_tile_gpu potrf_tile
#endif
#if !GEMM_SMP
#define gemm_tile_gpu gemm_tile
#endif
#if !TRSM_SMP
#define trsm_tile_gpu trsm_tile
#endif
#if !SYRK_SMP
#define syrk_tile_gpu syrk_tile
#endif


void laset_ (char * UPLO, int * M, int * N, REAL * ALPHA, REAL * BETA, REAL * A, int * LDA);

void gemm_ (const char *transa, const char *transb, int *l, int *n, int *m, REAL *alpha,
             const void *a, int *lda, void *b, int *ldb, REAL *beta, void *c, int *ldc);

REAL lange_ (char *norm, int *m, int *n, REAL *a, int *lda, REAL *work);

void larnv_ (int *idist, int *iseed, int *n, REAL *x);

void potrf_( char* uplo, int* n, REAL* a, int* lda, long* info );


float get_time();

int ISEED[4] = {0,0,0,1};
int IINFO;
int intONE=1;

//void gpu_spotf2_var1_( char *, int*, unsigned int*, int *, int * );
void gpu_spotrf_var1_( char *, int*, unsigned int*, int *, int *, int * );

void cholesky(REAL *Alin, REAL** Ah, int ts, int nt);

void add_to_diag_hierarchical (REAL ** matrix, int ts, int nt, REAL alpha)
{
    int i;

    for (i = 0; i < nt * ts; i++)
        matrix[(i/ts) * nt + (i/ts)][(i%ts) * ts + (i%ts)] += alpha;
}

void add_to_diag(REAL * matrix, int n, REAL alpha)
{
    int i;

    for (i = 0; i < n; i++)
        matrix[ i + i * n ] += alpha;
}

double gtod_ref_time_sec = 0.0;

float get_time()
{
    double t, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    // If this is the first invocation of through dclock(), then initialize the
    // "reference time" global variable to the seconds field of the tv struct.
    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double) tv.tv_sec;

    // Normalize the seconds field of the tv struct so that it is relative to the
    // "reference time" that was recorded during the first invocation of dclock().
    norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

    // Compute the number of seconds since the reference time.
    t = norm_sec + tv.tv_usec * 1.0e-6;

    return (float) t;
}


//--------------------------- check results --------------------

REAL sckres (int n, REAL * A, int lda, REAL * L, int ldl)
{
    REAL zero = 0.0;
    REAL minus_one = -1.0;
    int nminus_one = n - 1;
    REAL one = 1.0;

    REAL nrm, res;
    REAL dummy = 9;

    char NORM = '1';
    char T = 'T';
    char N = 'N';
    char U = 'U';

    res = 0;
    nrm = 0;

    nrm = lange_(&NORM, &n, &n, A, &lda, &dummy);

    laset_(&U, &nminus_one, &nminus_one, &zero, &zero, &L[ldl], &ldl);
    gemm_(&N, &T, &n, &n, &n, &minus_one, L, &ldl, L, &ldl, &one, A, &lda);
    REAL nrm2 = lange_(&NORM, &n, &n, A, &lda, &dummy);

    res = nrm2 / nrm;
    return res;
}


//----------------------------------------------------------------------------------
//			 Changes in data storage
//----------------------------------------------------------------------------------

void print_linear_matrix(int n, REAL *matrix)
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%g ", matrix[i*n + j]);
        }
        printf("\n");
    }
}

void read_matrix(char * filename, int n, int ts, REAL *matrix, REAL *checksum)
{
    int i = 0;
    FILE * matrix_file = NULL;
    if (filename != NULL) {
        matrix_file = fopen(filename, "r");
        if (!matrix_file) printf("Error opening matrix file: %s\n", strerror(errno));
    }

#ifdef DOUBLE_PREC
    // Reading file does not work for double precision
    if (matrix_file != 0) fclose(matrix_file);
    matrix_file = 0;
#endif
    
    if (matrix_file != 0) {
    
        while ((i < n*n) && (fscanf(matrix_file, "%g", &matrix[i]) != EOF)) {
            i++;
        }

        // Finished reading matrix; read checksum
        if (fscanf(matrix_file, "%g", checksum) == EOF) {
            printf("Invalid matrix file: could not read checksum\n");
            *checksum = 0.0;
            //exit(1);
        }
        fclose(matrix_file);
    }
    else {
#ifdef DOUBLE_PREC
        printf("Matrix file not valid for double precision, initializing matrix with random values\n");
#else
        printf("Matrix file not found, initializing matrix with random values\n");
#endif

        for (i = 0; i < n*n; i+=n) {
            larnv_(&intONE, &ISEED[0], &n, &matrix[i]);
        }

        int j;
        for (i=0; i<n; i++) {
            for (j=0; j<n; j++) {
                matrix[j*n + i] = matrix[j*n + i] + matrix[i*n + j];
                matrix[i*n + j] = matrix[j*n + i];
            }
        }

        add_to_diag(matrix, n, (REAL) n);

        *checksum = 0.0;
    }
}

#if CONVERT_TASK
#pragma omp target device (smp) copy_deps
#pragma omp task in([N*(ts-1)+ts]Alin) out([ts*ts]A) 
static void gather_block(int N, int ts, REAL *Alin, REAL *A)
{
    int i, j;

    for (i = 0; i < ts; i++)
       for (j = 0; j < ts; j++) {
          A[i*ts + j] = Alin[i*N + j];
       }
}
#endif

#if CONVERT_TASK
#pragma omp target device (smp) copy_deps
#pragma omp task in([ts*ts]A) out([N*(ts-1)+ts]Alin)
static void scatter_block(int N, int ts, REAL *A, REAL *Alin)
{
    int i, j;

    for (i = 0; i < ts; i++)
       for (j = 0; j < ts; j++) {
          Alin[i*N + j]= A[i*ts + j];
       }
}
#endif

// static void convert_to_blocks(int ts, int DIM, int N, REAL (*Alin)[N], REAL *(*A)[DIM])
static void convert_to_blocks(int ts, int DIM, int N, REAL Alin[N][N], REAL *A[DIM][DIM])
{
#if CONVERT_TASK
   int i, j;

   for (i = 0; i < DIM; i++)
     for (j = 0; j < DIM; j++) {
        nanos_current_socket( j % NUM_NODES );
        gather_block ( N, ts, &Alin[i*ts][j*ts], A[i][j]);
     }
#else
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            //A[j/NB][i/NB][(i%NB)*NB+j%NB] = Alin[i*N+j];
            A[j/ts][i/ts][(j%ts)*ts + i%ts] = Alin[j][i];
        }
    }
#endif
}

// static void convert_to_linear(int ts, int DIM, int N, REAL *(*A)[DIM], REAL (*Alin)[N])
static void convert_to_linear(int ts, int DIM, int N, REAL *A[DIM][DIM], REAL Alin[N][N])
{
#if CONVERT_TASK
   int i, j;

   for (i = 0; i < DIM; i++)
     for (j = 0; j < DIM; j++) {
        nanos_current_socket( j % NUM_NODES );
        CHECK_BLOCK_NOT_NULL(A[i][j])
        scatter_block ( N, ts, A[i][j], (REAL *) &Alin[i*ts][j*ts]);
     }
#else
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            Alin[j][i] = A[j/ts][i/ts][(j%ts)*ts + i%ts];
            //Alin[i*N + j] = A[i/NB][j/NB][(j%NB)*NB + i%NB];
        }
    }
#endif
}

#if CONVERT_REQUEST
static REAL * malloc_block (int ts)
{
   REAL *block;
   block = (REAL *) my_malloc(ts * ts * sizeof(REAL));
#if NANOS_API_COPIES_API >= 1004
   #pragma omp register ([ts*ts]block)
#endif

   if ( block == NULL ) {
      printf( "ALLOCATION ERROR (Ah block of %d elements )\n", ts );
      exit(-1);
   }

   return block;
}
#endif

//----------------------------------------------------------------------------------
//			 TASKS FOR CHOLESKY
//----------------------------------------------------------------------------------

#pragma omp target device (cuda) copy_deps
#pragma omp task inout([NB*NB]A) priority(100000)
void potrf_tile_gpu(REAL *A, int NB)
{
    char L = 'L';
    // Performing Cholesky on GPU
    int INFO;

    cudaStream_t stream = nanos_get_kernel_execution_stream();
    cublasSetKernelStream(stream);


    cublasHandle_t handle = nanos_get_cublas_handle();
    gpu_potrf( handle, L, NB, A, NB, &INFO );
}

#pragma omp target device (cuda) copy_deps
#pragma omp task in([NB*NB]A, [NB*NB]B) inout([NB*NB]C) priority(1)
void gemm_tile_gpu(REAL  *A, REAL *B, REAL *C, unsigned long NB)
{
    unsigned char TR = 'T', NT = 'N';
    REAL DONE = 1.0, DMONE = -1.0;

    cudaStream_t stream = nanos_get_kernel_execution_stream();
    // new api:
    //cublasSetStream(nanos_get_cublas_handle(), stream);

#if CUDA_VERSION < 5000
    cublasSetKernelStream(stream);
    gpu_blas_gemm(NT, TR,
                  NB, NB, NB,
                  DMONE, A, NB,
                  B, NB,
                  DONE,
                  C, NB);
#else
    cublasHandle_t handle = nanos_get_cublas_handle();
    cublasSetStream(handle, stream);
    gpu_blas_gemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                  NB, NB, NB,
                  &DMONE, A, NB,
                  B, NB,
                  &DONE,
                  C, NB);
#endif

}

#pragma omp target device (cuda) copy_deps
#pragma omp task in([NB*NB]T) inout([NB*NB]B) priority(priority)
void trsm_tile_gpu(REAL *T, REAL *B, unsigned long NB, unsigned priority)
{
    char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    REAL DONE = 1.0;

    // Performing STRSM on GPU

    cudaStream_t stream = nanos_get_kernel_execution_stream();

#if CUDA_VERSION < 5000
    cublasSetKernelStream(stream);
    gpu_blas_trsm(RI, LO, TR, NU, NB, NB,
                  DONE, T, NB, B, NB );
#else
    cublasHandle_t handle = nanos_get_cublas_handle();
    cublasSetStream(handle, stream);
    gpu_blas_trsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                  NB, NB,
                  &DONE, T, NB, B, NB );
#endif
}

#pragma omp target device (cuda) copy_deps
#pragma omp task in([NB*NB]A) inout([NB*NB]C) priority(priority)
void syrk_tile_gpu(REAL *A, REAL *C, long NB, unsigned priority)
{
    unsigned char LO = 'L', NT = 'N';
    REAL DONE = 1.0, DMONE = -1.0;


    cudaStream_t stream = nanos_get_kernel_execution_stream();

    // Performing SSYRK on GPU
#ifndef CUDA_VERSION
#error Cuda version not defined
#endif
#if CUDA_VERSION < 5000
    cublasSetKernelStream(stream);
    gpu_blas_syrk(LO, NT, NB, NB,
                DMONE, A, NB, DONE, C, NB );
#else
    cublasHandle_t handle = nanos_get_cublas_handle();
    cublasSetStream(nanos_get_cublas_handle(), stream);
    gpu_blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, NB, NB,
                &DMONE, A, NB, &DONE, C, NB );
#endif
}



//----------------------------------------------------------------------------------
//			 END TASKS FOR CHOLESKY
//----------------------------------------------------------------------------------

void cholesky(REAL *Alin, REAL** Ah, int ts, int nt)
{
    int i, j, k;

#if CONVERT_REQUEST
    int N = nt * ts;
#endif

    for (k = 0; k < nt; k++) {
        nanos_current_socket( k % NUM_NODES );
        TRY_GATHER_BLOCK(N, ts, &Alin[k*ts*N + k*ts], Ah[k*nt + k])
        // Diagonal Block factorization and panel permutations
        CALL_POTRF_TILE(Ah[k*nt + k], ts)

        // Triangular systems
        for (i = k + 1; i < nt; i++) {
            nanos_current_socket( i % NUM_NODES );
            TRY_GATHER_BLOCK(N, ts, &Alin[k*ts*N + i*ts], Ah[k*nt + i])
            trsm_tile(Ah[k*nt + k], Ah[k*nt + i], ts, (nt-i)+10);
        }
        // update trailing matrix
        for (i = k + 1; i < nt; i++) {
            nanos_current_socket( i % NUM_NODES );
            for (j = k + 1; j < i; j++) {
                TRY_GATHER_BLOCK(N, ts, &Alin[k*ts*N + j*ts], Ah[k*nt + j])
                TRY_GATHER_BLOCK(N, ts, &Alin[i*ts*N + j*ts], Ah[j*nt + i])
                gemm_tile(Ah[k*nt + i], Ah[k*nt + j], Ah[j*nt + i], ts);
            }
            TRY_GATHER_BLOCK(N, ts, &Alin[i*ts*N + i*ts], Ah[i*nt + i])
            syrk_tile(Ah[k*nt + i], Ah[i*nt + i], ts, (nt-i)+10);
        }
    }
}

//--------------------------- MAIN --------------------
int main(int argc, char* argv[])
{

    float t1, t2;
    REAL* matrix;
    REAL* original_matrix = NULL;
    REAL ** Ah; 		// Ahdow matrix
    REAL checksum;
    REAL res = -1;
    REAL sum = 0.0;
    int i, j, n, check_result;
    int ts, nt;

    if ( argc != 5 && argc != 4) {
        printf( "cholesky size block_size check_result [matrix_file]\n" );
        exit( -1 );
    }

    n = atoi(argv[1]);
    ts = atoi(argv[2]);
    check_result = atoi(argv[3]);

    if (ts > n) {
        printf( "Error: block size is bigger than matrix size\n");
        exit(-1);
    }
    
    nanos_get_num_sockets( &NUM_NODES );
    printf( "Running with %d nodes\n", NUM_NODES );

    // Allocate matrix
    matrix = (REAL *) malloc(n * n * sizeof(REAL));
#if NANOS_API_COPIES_API >= 1004
    #pragma omp register ([n*n]matrix)
#endif
    if (matrix == NULL) {
        printf("ALLOCATION ERROR\n");
        exit(-1);
    }

    read_matrix((argc == 5 ? argv[4] : NULL), n, ts, matrix, &checksum);

    // Allocate matrix
    if (check_result) {
        original_matrix = (REAL *) malloc(n * n * sizeof(REAL));
        if (original_matrix == NULL) {
            printf("ALLOCATION ERROR\n");
            exit(-1);
        }
    }

    nt = n / ts;

    // Allocate blocked matrix
    Ah = (REAL **) malloc(nt * nt * sizeof(REAL *));
    if (Ah == NULL) {
        printf("ALLOCATION ERROR (Ah)\n");
        exit(-1);
    }

    for (j = 0; j < nt * nt; j++) {
        Ah[j]=(REAL *) my_malloc(ts * ts * sizeof(REAL));
        if (Ah[ j ] == NULL) {
            printf("ALLOCATION ERROR (Ah[%d] )\n", j);
            exit(-1);
        }
    }

    if (check_result) {
        for (i = 0; i < n * n; i++ ) {
            original_matrix[i] = matrix[i];
        }
    }

    // gmiranda: too early to compute time
    //t1 = get_time();
#if !CONVERT_REQUEST
    convert_to_blocks(ts, nt, n, (REAL(*)[n]) matrix, (REAL* (*)[nt]) Ah);
// wait before converting to blocks, don't compute the time until it's done
#pragma omp taskwait noflush
#endif
    t1 = get_time();
    cholesky(matrix, Ah, ts, nt);
#if !CONVERT_TASKS
// If convert_to_linear is sequential, wait for all tasks before going on
#pragma omp taskwait
#endif
  
    t2 = get_time() - t1;
    convert_to_linear(ts, nt, n, (REAL* (*)[nt]) Ah, (REAL (*)[n]) matrix);
#pragma omp taskwait
    //t2 = get_time() - t1;

    for (i = 0; i < n * n; i++) {
        sum += matrix[i];
    }

    if (check_result) {
        res = sckres(n, original_matrix, n, matrix, n);
        free(original_matrix);
    }

    // Free blocked matrix
    for (j = 0; j < nt * nt; j++) {
        my_free(Ah[j]);
    }
    free(Ah);
    free(matrix);

    float time = t2;
    float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

    int check = (check_result && (res > accepted_error)) || (!check_result && (fabs(fabs(sum) - fabs(checksum)) > accepted_error));

    // Print configuration
    PRINT_PRECISION;
    printf("\tCONVERT_TASK %d\n\tCONVERT_REQUEST %d\n\tPOTRF_SMP %d\n\tPOTRF_NESTED %d\n\tUSE_PRIORITY %d\n\tUSE_PINNED %d\n\tUSE_IMPLEMENTS %d\n", CONVERT_TASK, CONVERT_REQUEST, POTRF_SMP, POTRF_NESTED, USE_PRIORITY, USE_PINNED, USE_IMPLEMENTS);

    // Print results
    printf( "============ CHOLESKY RESULTS ============\n" );
    printf( "  matrix size:                      %dx%d\n", n, n);
    printf( "  block size:                       %dx%d\n", ts, ts);
    printf( "  computation time (in seconds):    %f\n", time);
    printf( "  performance GFLOPS:               %f\n", gflops);
    printf( "  checksum computed:           %10.10g\n", sum);
    printf( "  checksum expected:           %10.10g\n", checksum);
    if (check_result) {
        printf("  Residual: %g\n", res);
    }
    if (check) {
        printf("  Error checking failed\n");
        printf( "==========================================\n" );
	return 1;
    }
    else {
        //printf("  Error is within acceptable tolerance :)\n");
        printf("  Verification: Ok\n");
        printf( "==========================================\n" );
    }


    return 0;
}
