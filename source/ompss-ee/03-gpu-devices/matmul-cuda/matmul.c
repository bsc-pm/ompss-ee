#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include "matmul_kernel.cuh"

#include "driver.h"

#ifdef DP
#define REAL double
#else
#define REAL float
#endif


const int NB = BSIZE;
void matmul( int m, int l, int n, int mDIM, int lDIM, int nDIM, REAL **tileA, REAL **tileB,
             REAL **tileC )
{	
	int i, j, k;
	for(i = 0;i < mDIM; i++){
		for (j = 0; j < nDIM; j++){ 
			for (k = 0; k < lDIM; k++){
				Muld(tileA[i*lDIM+k], tileB[k*nDIM+j],NB,NB, tileC[i*nDIM+j],NB);
			}
		}
	}
}


