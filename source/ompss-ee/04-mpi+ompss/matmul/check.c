#include "bsize.h"
#include "matmul.h"
#include<stdio.h>
#include<math.h>
#include<float.h>

//#pragma omp target device (smp)  copy_deps // copy_in ([ts]pb)
//#pragma omp task input(pb[0:ts-1]) concurrent (*ok)
void check_block (int m, int n, double (*pb)[n], double value, double eps, int *ok)
{
   int i, j;
   int lok=0;

   for(i=0;i<m;i++){
      for(j=0;j<n;j++){
         if ( fabs( value - (pb[i][j]) ) > eps ) {
            lok++;
         }
      }
   }

   if (lok >0) *ok+=lok;  //does not matter if no mx
}

int check(int m, int n, double (*C)[n], double tvalue)
{
   double eps;
   int    i, j, ok = 0;

   eps = 2.0*m*m*DBL_EPSILON;

   for(i=0;i<1;i++){
      for(j=0;j<1;j++){
         check_block( m, n, C, tvalue, eps, &ok);
      }
   }

   return( ok );
}

