#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#include "driver.h"

#ifdef DP
#define REAL double
#else
#define REAL float
#endif


double   cclock( void );
int      check( int nrep, int m, int l, int n, int mDIM, int nDIM, REAL **c );
void     gendat(int, int, int, int, int, int, REAL **, REAL **, REAL **);
void     matmul( int, int, int, int, int, int, REAL **a, REAL **b, REAL **c );
void     prthead( void );
void     prtspeed( int, int, int, int, double, int, unsigned long );

int calcdim(int x)
{
        int dimval;
        if(x%BSIZE != 0)
                dimval = x/BSIZE + 1;
        else
                dimval = x/BSIZE;

        return dimval;
}
	
int main()
{ 
	int      lda, m, l, n;
	int      mDIM, lDIM, nDIM;
	int      ok, nrep;
	unsigned long nops;
	int      i,k,j,o;
	REAL   **a, **b, **c;
	double   time;
	FILE     *inl;
// ------------------------------------------------------------------------	

	inl = fopen( "test.in", "r" );
	if (inl == 0) {
		printf("No input file 'test.in' found.\n");
		exit(1);
	}



	while( ( fscanf( inl, "%d%d%d%d\n", &m, &l, &n, &nrep ) != EOF ) ){
		lda = l + 1;

		mDIM = calcdim(m);
		lDIM = calcdim(l);
		nDIM = calcdim(n);

		a = (REAL **)malloc( mDIM*lDIM*sizeof( REAL *) );
		b = (REAL **)malloc( lDIM*nDIM*sizeof( REAL *) );
		c = (REAL **)malloc( mDIM*nDIM*sizeof( REAL *) );
      
		for(i=0;i<mDIM*lDIM;i++)
			a[i] = (REAL *)malloc(BSIZE*BSIZE*sizeof(REAL));

		for(i=0;i<lDIM*nDIM;i++)
			b[i] = (REAL *)malloc(BSIZE*BSIZE*sizeof(REAL));

		for(i=0;i<mDIM*nDIM;i++)
			c[i] = (REAL *)malloc(BSIZE*BSIZE*sizeof(REAL));


		gendat( mDIM, lDIM, nDIM, m, l, n, a, b, c );

		time = cclock();
		for( i = 0; i < nrep; i++ ){
			matmul( m, l, n, mDIM, lDIM, nDIM, a, b, c ); 
		}
#pragma omp taskwait
		time = cclock() - time;
		ok   = check( nrep, m, l, n, mDIM, nDIM, c);

		time = time/nrep;
		nops  = (unsigned long) 2*m*l*n;
		prtspeed( m, l, n, BSIZE, time, ok, nops );

		for(i=0;i<mDIM*lDIM;i++)
			free( a[i] );

		for(i=0;i<lDIM*nDIM;i++)
			free( b[i] );
  
		for(i=0;i<mDIM*nDIM;i++)
			free( c[i] );

		free( a ); free( b ); free( c );
	}


	//printf( "----------------------------------------------------------\n" );
}
