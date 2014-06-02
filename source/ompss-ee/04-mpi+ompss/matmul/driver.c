#include "bsize.h"
#include "matmul.h"
#include "layouts.h"
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


double   cclock( void );
int      check(int m, int n, double (*C)[n], double tvalue);
void     clear_mat(int DIM, int BS, double* Mat[DIM][DIM]);
void     gendat(int m, int n, double (*A)[m], double (*B)[n],  double (*C)[n]);
void     prthead( int nodes );
void     prtspeed( int, int, int, double, int, unsigned long );

 
// MPI info. Global Variables. Invariant during whole execution
int me;
int nodes;   


int main(int argc, char **argv)
{ 
   MPI_Comm comm = MPI_COMM_WORLD;
   int      root=0;

   int      m, n;
   int      DIM;
   int      BS = BSIZE;
   int      gok, ok, nrep;
   unsigned long nops;
   int      i,k,j,o;
   double   gtime, time;
   FILE     *inl;
   int      r;
// ------------------------------------------------------------------------	

   MPI_Init( &argc, &argv );
   MPI_Comm_rank( comm, &me );
   MPI_Comm_size( comm, &nodes );

   if ( me == 0 ) {
      prthead( nodes );
      inl = fopen( "test.in", "r" );
      if (inl == 0) 
         printf("No input file 'test.in' found.\n");

   }

   MPI_Bcast(&inl, 1, MPI_INT, root, comm);
   if (inl == 0) {
      MPI_Finalize();
      exit(1);
   }

   if (me == 0) r = fscanf( inl, "%d%d%d\n", &m, &BS, &nrep );
      MPI_Bcast(&r, 1, MPI_INT, root, comm);
      MPI_Bcast(&m, 1, MPI_INT, root, comm);
      MPI_Bcast(&BS, 1, MPI_INT, root, comm);
      MPI_Bcast(&nrep, 1, MPI_INT, root, comm);

   while ( r != EOF) {

      if (m%BS != 0) {
	 if ( me==0) { printf ("Only full blocks supported\n");}
         MPI_Finalize();
         exit(-1);
      }

      n = m/nodes;

      int i,j;
      double (*a)[m];              // a: n rows of size m
      double (*b)[n], (*c)[n];    // b and c: m rows of size n


// **************** Allocate  matrices **************************
      a = (double (*)[m]) malloc(m*n*sizeof(double));
      b = (double (*)[n]) malloc(m*n*sizeof(double)); 
      c = (double (*)[n]) malloc(m*n*sizeof(double));

      gendat( m, n, a, b, c);

      // to show if somebody else is running on our nodes
      //system ("ps -emo psr,bnd,user,args");

      time = MPI_Wtime();

      for( i = 0; i < nrep; i++ ){
         matmul( m, n, a, b, c ); 
      }

#pragma omp taskwait
      time = MPI_Wtime() - time;

      ok = check( m, n, c, (double)nrep*m);
      MPI_Reduce( &ok, &gok, 1, MPI_INT, MPI_LAND, 0, comm );

      time = time/nrep;
      MPI_Reduce( &time, &gtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
      nops  = (unsigned long) 2*m*m*m;
      if (me == 0) prtspeed( m, m, m, gtime, gok, nops );

      free(a);
      free(b);
      free(c);

      if (me == 0) r = fscanf( inl, "%d%d%d\n", &m, &BS, &nrep );
      MPI_Bcast(&r, 1, MPI_INT, root, comm);
      MPI_Bcast(&m, 1, MPI_INT, root, comm);
      MPI_Bcast(&BS, 1, MPI_INT, root, comm);
      MPI_Bcast(&nrep, 1, MPI_INT, root, comm);
   }

   MPI_Barrier(comm);

   MPI_Finalize();

   if ( me == 0 ) printf( "----------------------------------------------------------\n" );

	return ok;
}
