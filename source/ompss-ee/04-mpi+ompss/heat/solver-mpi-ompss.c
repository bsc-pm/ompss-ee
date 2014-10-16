#include "heat.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 8
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double (*um)[sizey] = u, (*utmpm)[sizey] = utmp;

    double diff, sum=0.0;
    int nbx, bx, nby, by;
    int inf_i, sup_i, inf_j, sup_j;
    int up, left, right, down;
  
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    for (int ii=1; ii<sizex-1; ii+=bx)
        for (int jj=1; jj<sizey-1; jj+=by) {
	        inf_i = ii; inf_j=jj; 
	        sup_i = (ii+bx)<sizex-1 ? ii+bx : sizex-1; 	
	        sup_j = (jj+bx)<sizey-1 ? jj+by : sizey-1; 

	        up = (ii-bx)<0? 0 : ii-bx;
	        left = (jj-by)<0 ? 0 : jj - by; 
	        down = (ii+bx)> sizex-1? sizex-1 : ii+bx;
            right = (jj+by)>sizey-1 ? sizey - 1 : jj+by; 

            #pragma omp task in (um[up][jj], um[ii][left], um[down][jj], um[ii][right]) out (utmpm[ii][jj])  concurrent(sum) firstprivate(ii, left, down, jj, right, up) label(compute_jacobi)
            {
	            double local_sum=0.0;
                for (int i=inf_i; i<sup_i; i++) 
                    for (int j=inf_j; j<sup_j; j++) {
                        utmpm[i][j]= 0.25 * (um[ i][j-1 ]+  // left
			                 um[ i][(j+1) ]+  // right
	                         um[(i-1)][ j]+  // top
	                         um[ (i+1)][ j ]); // bottom
                        diff = utmpm[i][j] - um[i][j];
                        local_sum += diff * diff; 
                    }

                #pragma omp atomic
	            sum += local_sum;
            }
	    }

#pragma omp taskwait
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
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
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

    return sum;
}

