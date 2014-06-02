#include <stdio.h>

void prthead( int nodes )
{
   printf( "matmul: Matrix-matrix multiply test C(m,n) = A(m,l)*B(l,n)\n" );
   printf ("Number of MPI processes: %d\n", nodes);
   printf( "----------------------------------------------------------\n" );
   printf( "   Problem size  |              |             |     |\n" );
   printf( "  m  |  l  |  n  |  Time (s)    | (Gflop/s)   | OK? |\n" );
   printf( "----------------------------------------------------------\n" );
   fflush(stdout);

}
