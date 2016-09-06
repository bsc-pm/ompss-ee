#include<stdio.h>
#include<math.h>
#include<float.h>

#include "driver.h"

#ifdef DP
#define REAL double
#else
#define REAL float
#endif

//#define BSIZE 1024

int check( int nrep, int m, int l, int n, int mDIM, int nDIM, REAL **c/*[][nDIM*BSIZE] */)
{
	double eps, tvalue = (double)l;
	int    i, j, k, o, ok = 0;

	eps = 2.0*l*l*DBL_EPSILON;
	int perfectM = m / BSIZE;
	int perfectN = n / BSIZE;

	int leftOutM = m % BSIZE;
	int leftOutN = n % BSIZE;

	for(i=0;i<mDIM;i++){
		for(j=0;j<nDIM;j++){
			for(k=0;k<BSIZE;k++){
				for(o=0;o<BSIZE;o++){
					if( i == mDIM-1 && mDIM > perfectM && k >= leftOutM )
						break;
					else if( j == nDIM-1 && nDIM > perfectN && o >= leftOutN )
						break;
					else {
						if ( fabs( tvalue - (c[i*nDIM+j][k*BSIZE+o]/nrep) ) > eps ) {
							ok++;
							//printf("Bad result at [%d][%d] : expected %f but found %f\n", i*nDIM+j, k*BSIZE+o, tvalue, c[i*nDIM+j][k*BSIZE+o]);
						}
					}
				}
			}
		}
	}

	return( ok );
}

