#include "heat.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 8
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;

#pragma omp parallel for private(diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
#pragma omp parallel for private(lsw,unew,diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
#pragma omp parallel for private(lsw,unew,diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;

    for (int diag = 0; diag < nbx; diag++)
#pragma omp parallel for private(unew,diff) reduction(+:sum)
      for (int I=0; I<=diag; I++) {
            int ii = I;
            int jj = diag - I;
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (    u[ i*sizey        + (j-1) ]+  // left
                                      u[ i*sizey        + (j+1) ]+  // right
                                      u[ (i-1)*sizey    + j     ]+  // top
                                      u[ (i+1)*sizey    + j     ]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff;
                    u[i*sizey+j]=unew;
                }
         }

    for (int diag = nbx-1; diag >= 0; diag--)
#pragma omp parallel for private(unew,diff) reduction(+:sum)
      for (int I=nbx-1; I>=nbx-diag; I--) {
            int ii = nby-I;
            int jj = nby-ii;
            ii += nbx-1-diag;
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (    u[ i*sizey        + (j-1) ]+  // left
                                      u[ i*sizey        + (j+1) ]+  // right
                                      u[ (i-1)*sizey    + j     ]+  // top
                                      u[ (i+1)*sizey    + j     ]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff;
                    u[i*sizey+j]=unew;
                }
         }

    return sum;
}

