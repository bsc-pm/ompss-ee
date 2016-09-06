#include <stdio.h>

#define max(a,b)( ((a) > (b)) ? (a) : (b) )

void prtspeed( int m, int l, int n, double time, int ok, unsigned long nops )
{
   double speed;
// -----------------------------------------------------------------
   //speed = 1.0e-9*2*m*l*n/max( time, 1.0e-9 );
   speed = 1.0e-9*nops/time;

//   printf( "%4d |%4d | %4d| %11.4lf   | %11.4lf | ", m, l, n, time, speed );
   printf( "%d\t%d\t%d\t%.4lf\t   %.4lf     ", m, l, n, time, speed );
   if ( ok == 0 )
      printf( " T |\n" );
   else
      printf( " F (%d)|\n", ok );
//   printf( "nops = %lu; m = %d; l = %d; n = %d\n", nops, m, l, n );

   fflush(stdout);
}
