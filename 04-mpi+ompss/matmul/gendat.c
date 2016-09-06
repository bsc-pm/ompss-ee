#include "matmul.h"
#include "layouts.h"
#include <stdio.h>


//#pragma omp target device (smp) copy_deps
//#pragma omp task output(A)
void init_tile (int m, int n, double (*A)[n], double Value )
{
   int i, j; 

   for( i = 0; i < m; ++i  ) 
      for( j = 0; j < n; ++j ) 
         A[i][j] = Value;

}

void     gendat(int m, int n, double (*A)[m], double (*B)[n],  double (*C)[n])
{        
   int i,j;
   double Value;

   for( i = 0; i < 1; ++i )
      for( j = 0; j < 1; ++j ) {
         Value = 1.0;
         init_tile( n, m, A, Value);
         Value = 1.0;
         init_tile( m, n, B, Value);
	 Value = 0.0;
         init_tile( m, n, C, Value);
      }
}
