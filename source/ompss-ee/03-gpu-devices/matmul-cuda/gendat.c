#include <stdio.h>

#include "driver.h"

#define INIT_TASKS 0

#ifdef DP
#define REAL double
#else
#define REAL float
#endif

//#define BSIZE 1024

#if !INIT_TASKS
void gendat(int mDIM, int lDIM, int nDIM, int m, int l, int n, REAL **tileA, REAL **tileB, REAL **tileC)
{ 
	int i,j,k,y;
	REAL currentValue;

	int perfectM = m / BSIZE;
	int perfectL = l / BSIZE;
	int perfectN = n / BSIZE;

	int leftOutM = m % BSIZE;
	int leftOutL = l % BSIZE;
	int leftOutN = n % BSIZE;

	for( i = 0; i < mDIM; ++i )
		for( j = 0; j < lDIM; ++j )
			for( k = 0; k < BSIZE; ++k  )
			{
				currentValue = j*BSIZE;
				for( y = 0; y < BSIZE; ++y )
				{
					if( i == mDIM-1 && mDIM > perfectM && k >= leftOutM )
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = 0.0;
					else if( j == lDIM-1 && lDIM > perfectL && y >= leftOutL )
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = 0.0;
					else
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = ++currentValue;
				}
			}

	for( i = 0; i < lDIM; ++i )
		for( j = 0; j < nDIM; ++j )
		{
			currentValue = (i*BSIZE) + 1;
			for( k = 0; k < BSIZE; ++k,  currentValue += 1)
				for( y = 0; y < BSIZE; ++y )
				{
					if( i == lDIM-1 && lDIM > perfectL && k >= leftOutL )
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 0.0;
					else if( j == nDIM-1 && nDIM > perfectN && y >= leftOutN )
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 0.0;
					else
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 1.0 / currentValue;
				}
		}

	for( i = 0; i < lDIM; ++i )
		for( j = 0; j < nDIM; ++j )
		{
			for( k = 0; k < BSIZE; ++k)
				for( y = 0; y < BSIZE; ++y )
				{
					tileC[ i*nDIM + j ][ k*BSIZE+y ] = 0.0;
				}
		}

}
#else
//not working in OCL/CUDA yet
void gendat(int mDIM, int lDIM, int nDIM, int m, int l, int n, REAL **tileA, REAL **tileB, REAL **tileC)
{        
	int i,j;

	for( i = 0; i < mDIM; ++i ) {
		for( j = 0; j < lDIM; ++j ) {
			REAL *_c;
			REAL *_a;
			REAL *_b;
			_c = tileC[i * mDIM + j];
			_a = tileA[i * mDIM + j];
			_b = tileB[j * mDIM + i];
			initMatrix(_a,_b,_c,m);
		}
	}
}
#endif
