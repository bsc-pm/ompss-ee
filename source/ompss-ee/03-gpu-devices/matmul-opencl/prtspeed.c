#include <stdio.h>

#define max(a,b)( ((a) > (b)) ? (a) : (b) )
#if 0
void prtspeed( int m, int l, int n, double time, int ok, unsigned long nops )
{
   double speed;
// -----------------------------------------------------------------
   //speed = 1.0e-9*2*m*l*n/max( time, 1.0e-9 );
   speed = 1.0e-9*nops/max( time, 1.0e-9 );

   printf( "%4d |%4d | %4d| %11.4lf   | %11.4lf | ", m, l, n, time, speed );
   if ( ok == 0 )
      printf( " T |\n" );
   else
      printf( " F (%d)|\n", ok );
}
#else

void prtspeed( int m, int l, int n, int nb, double time, int ok, unsigned long nops )
{
	double speed = 1.0e-9*nops/time;
	printf("Matrix size: %dx%d\n", m, n);
	printf("Block size: %dx%d\n", nb, nb);
#ifdef DP
	printf("Precision type: Double\n");
#else
	printf("Precision type: Simple\n");
#endif

	printf("  GFLOPS : %.4lf\n", speed);
	printf("  computation time (in seconds): %.4lf\n", time);
	if ( ok == 0 ) {
		printf("  Verification: Ok\n");
	} else {
		printf("  Verification: Failed  (%d)\n", ok);
	}
}
#endif
