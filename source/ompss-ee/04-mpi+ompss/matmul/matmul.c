#include "bsize.h"
#include "matmul.h"
#include "layouts.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <mpi.h>


// MPI info. Global Variables. Invariant during whole execution
extern int me;
extern int nodes;   

void matmul ( int m, int n, double (*A)[m], double (*B)[n], double (*C)[n] )
{
    double (*a)[m];
    double (*rbuf)[m];
    double (*orig_rbuf)[m];
    void   *ptmp;
    int up, down;
    int i;
    int it;
    int tag = 1000;
    int size = m*n;
    MPI_Status stats;

    orig_rbuf = rbuf = (double (*)[m])malloc(m*n*sizeof(double));
    if (nodes >1) {
       up =   me<nodes-1 ? me+1:0;
       down = me>0       ? me-1:nodes-1;
    } else {
       up = down = MPI_PROC_NULL;
    }
    
    a=A;
    i = n*me;  // first C block (different for each process)
    size = m*n;


    for( it = 0; it < nodes; it++ ) {

      #pragma omp task in (a[0:n-1], B[0:m-1]) inout (C[i:i+n-1][0:n-1]) firstprivate (n,m)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, m, 1.0, (double *)a, m, (double *)B, n, 1.0, (double *)&C[i][0], n);

      if (it < nodes-1) {
         #pragma omp task in (a[0:n-1]) out (rbuf[0:n-1]) inout(stats) firstprivate (size,m,n,tag,down,up)
         MPI_Sendrecv( a, size, MPI_DOUBLE, down, tag, rbuf, size, MPI_DOUBLE, up, tag, MPI_COMM_WORLD, &stats );
      }

      i = (i+n)%m;                 //next C block circular
      ptmp=a; a=rbuf; rbuf=ptmp;   //swap pointers
    }

#pragma omp taskwait
    free (orig_rbuf);
}
