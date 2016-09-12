#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

long N;
long CHUNK_SIZE;


static void initialize(long length, double data[length])
{
    for (long i = 0; i < length; i++) {
        data[i] = ((double)rand()/(double)RAND_MAX);
    }
}

double dot_product (long N, long CHUNK_SIZE, double A[N], double B[N])
{
    long actual_size;
    int j;
    double result;

    const long N_CHUNKS = N/CHUNK_SIZE + (N % CHUNK_SIZE != 0);
    double *C = malloc (N_CHUNKS*sizeof(double));

    result=0.0;
    j=0;
    for (long i=0; i<N; i+=CHUNK_SIZE) {

        actual_size = (N - i >= CHUNK_SIZE) ? CHUNK_SIZE : N - i;

        // OMPSS: What are the 2 inputs and the in/out data for this task ?
        #pragma omp task label( dot_prod ) firstprivate( j, i, actual_size ) in( A[i;actual_size], B[i; actual_size] ) inout( C[j;1] )
        {
            C[j]=0;
            for (long ii=0; ii<actual_size; ii++)
                C[j]+= A[i+ii] * B[i+ii];
        }

        // OMPSS: This task depends on an single element of C and will resultumulate the result on result.
        #pragma omp task label( increment ) firstprivate( j ) in( C[j;1] ) commutative( result )
        result += C[j];

        j++;
    }

    // OMPSS: We must make sure that all computations have ended before returning a value
    #pragma omp taskwait

    return(result);
}

int main(int argc, char **argv) {

    struct timeval start;
    struct timeval stop;
    unsigned long elapsed;
    double result;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector size in K> <chunk size in K> \n", argv[0]);
        return 1;
    }

    N = atol(argv[1]) * 1024L;
    CHUNK_SIZE = atol(argv[2]) * 1024L;

    double *A = malloc(N*sizeof(double));
    double *B = malloc(N*sizeof(double));

    initialize(N, A);
    initialize(N, B);

    gettimeofday(&start,NULL);
    result = dot_product (N, CHUNK_SIZE, A, B);

    gettimeofday(&stop,NULL);
    elapsed = 1000000 * (stop.tv_sec - start.tv_sec);
    elapsed += stop.tv_usec - start.tv_usec;
    printf ("Result of Dot product i= %le\n", result);
    printf("time (us): ");
    printf ("%lu;\n", elapsed);
}
