#include "bsize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <errno.h>

#include "layouts.h"

void clear_double( int n, REAL (*matrix)[n])
{
   int i,j;
   for (i=0; i<n; i++) {
      for (j=0; j<n; j++) 
         matrix[i][j] = 0.0;
   }
}


//----------------------------------------------------------------------------------
//		 Memory management and changes in data storage
//----------------------------------------------------------------------------------

void gather_block(int N, REAL (*Alin)[N], int I, int J, int ts, REAL (*A)[ts])
{
    int i, j;

    for (i = 0; i < ts; i++)
       for (j = 0; j < ts; j++) {
          A[i][j] = Alin[I*ts+i][J*ts+j];
       }
}

void scatter_block (int ts, REAL (*A)[ts], int N, REAL (*Alin)[N], int I, int J)
{
    int i, j;

    for (i = 0; i < ts; i++)
       for (j = 0; j < ts; j++) {
          Alin[I*ts+i][J*ts+j]= A[i][j];
       }
}

REAL * malloc_block (int ts)
{
   REAL *block;
   block = (REAL *) malloc(ts * ts * sizeof(REAL));

   if ( block == NULL ) {
      printf( "ALLOCATION ERROR (Ah block of %d elements )\n", ts );
      exit(-1);
   }

   return block;
}

void check_gather_block(int N, REAL (*Alin)[N], int I, int J, int ts, REAL **Ah )
{
  REAL (*p)[ts];
  if (*Ah == NULL) {
     p =  (REAL (*)[ts])malloc_block(ts); 
     *Ah = (REAL*) p;
  } else {
     p = (REAL (*)[ts])*Ah;
  }
  gather_block (N, Alin, I, J, ts, p);
}



void convert_to_blocks(int N, REAL (*Alin)[N], int DIM, REAL * (*Ah)[DIM])
{
   int i, j;
   int ts = N/DIM;
   if (N%DIM !=0) {printf ("Matrix size should be multiple of block size\n"); exit(-1);}

   for (i = 0; i < DIM; i++)
     for (j = 0; j < DIM; j++) {
        check_gather_block( N, (double (*)[N])Alin, i, j, ts, &Ah[i][j]);
     }
}

void convert_3_to_blocks(int N, REAL (*Alin)[N], REAL (*Blin)[N], REAL (*Clin)[N],
                         int DIM, REAL * (*Ah)[DIM],  REAL * (*Bh)[DIM],  REAL * (*Ch)[DIM])
{
   int i, j;
   int ts = N/DIM;
   if (N%DIM !=0) {printf ("Matrix size should be multiple of block size\n"); exit(-1);}

   for (i = 0; i < DIM; i++)
     for (j = 0; j < DIM; j++) {
        check_gather_block( N, (double (*)[N])Alin, i, j, ts, &Ah[i][j]);
        check_gather_block( N, (double (*)[N])Blin, i, j, ts, &Bh[i][j]);
        check_gather_block( N, (double (*)[N])Clin, i, j, ts, &Ch[i][j]);
     }
}

void convert_to_linear(int ts, int DIM, REAL * (*A)[DIM], int N, REAL Alin[N][N])
{
   int i, j;

   for (i = 0; i < DIM; i++)
     for (j = 0; j < DIM; j++) {
        if (A[i][j] != NULL) {
           scatter_block ( ts, (REAL (*)[ts]) A[i][j], 
                            N, (REAL (*)[N]) Alin, i, j);
           }
     }
}

REAL **  alloc_empty_block_matrix (int nt)
{
    int i;
    REAL ** Ah;

    Ah = (REAL **) malloc(nt * nt * sizeof(REAL *));
    if (Ah == NULL) {
        printf("ALLOCATION ERROR (Ah)\n");
        exit(-1);
    }
    for (i=0; i<nt*nt; i++) Ah[i]=NULL;

    return Ah;
}


REAL **  alloc_block_matrix (int nt, int ts)
{
    int i;
    REAL ** Ah;

    Ah = (REAL **) malloc(nt * nt * sizeof(REAL *));
    // printf ("allocated %d at %x\n", nt * nt * sizeof(REAL *), Ah);
    if (Ah == NULL) {
        printf("ALLOCATION ERROR (Ah)\n");
        exit(-1);
    }
    for (i=0; i<nt*nt; i++) {
      Ah[i]=malloc_block(ts);
      // printf ("allocated %d at %x\n", ts, Ah[i]);
   }

    return Ah;
}


void free_block_matrix ( int nt, REAL *A[nt][nt])
{
   int i, j;

   for (i=0; i<nt; i++)
      for(j=0; j<nt; j++)
         if(A[i][j] != NULL) {
            // printf ("freeing %x\n", A[i][j]);
            free(A[i][j]);
         }
   // printf ("freeing %x\n", A);
   free (A);
}
