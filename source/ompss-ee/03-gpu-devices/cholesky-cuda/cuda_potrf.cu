

#include "cuda_potrf.cuh"
// http://www.ast.cam.ac.uk/~stg20/cuda/cholesky/


const unsigned BLOCK_SIZE = 16;

#define MAT_POS(m,size,i,j) (m)[(i)*size+j]

// this is a small kernel that Cholesky decomposes the current "top left" 
// block of the matrix...

template<typename T>
__global__ void d_choldc_topleft(T *m,
				 int boffset, int matsize)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ T topleft[BLOCK_SIZE][BLOCK_SIZE+1];

    topleft[ty][tx]=MAT_POS( m, matsize, ty+BLOCK_SIZE*boffset, tx+BLOCK_SIZE*boffset );

    __syncthreads();

    T diagelem,fac;


// in this loop tx labels column, ty row
    for(int k=0;k<BLOCK_SIZE;k++)
    {
	__syncthreads();
	fac=1./sqrt(topleft[k][k]);
	__syncthreads();
	if ((ty==k)&&(tx>=k)) 
	{
	    topleft[ty][tx]=(topleft[ty][tx])*fac;
	}
	__syncthreads();

	if ((tx>=ty)&&(ty>k)) 
	{
	    topleft[ty][tx]=topleft[ty][tx]-topleft[k][ty]*topleft[k][tx]; 
	}

    }

    __syncthreads();


    if (tx>=ty) {
	MAT_POS( m, matsize, ty+BLOCK_SIZE*boffset, tx+BLOCK_SIZE*boffset)
	    =topleft[ty][tx];
    }

}

// this kernel updates the strip below the "topleft" block
template<typename T>
__global__ void d_choldc_strip(T *m,
			       int blockoffset, int matsize)
{

// +1 since blockoffset labels the "topleft" position
// and boff is the working position...
    int boffy=blockoffset;
    int boffx = blockIdx.x+blockoffset+1; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    
    __shared__ T topleftt[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ T workingmat[BLOCK_SIZE][BLOCK_SIZE+1];

// deliberately transposed...
    topleftt[tx][ty]=MAT_POS( m, matsize, ty+blockoffset*BLOCK_SIZE, tx+blockoffset*BLOCK_SIZE);

    workingmat[ty][tx]=
	MAT_POS( m, matsize,ty+boffy*BLOCK_SIZE, tx+boffx*BLOCK_SIZE);

    __syncthreads();

    // now we forward-substitute for the new strip-elements...
    // one thread per column (a bit inefficient I'm afraid)

    if(ty==0)
    {
	for (int k=0;k<BLOCK_SIZE;k++)
	{
	    T dotprod=0.;
	    for (int m=0;m<k;m++)
	    {
		dotprod+=topleftt[k][m]*workingmat[m][tx];
	    }
	    workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleftt[k][k];
	}
    }

    __syncthreads();

    MAT_POS( m, matsize, ty+blockoffset*BLOCK_SIZE, tx+boffx*BLOCK_SIZE)
	=workingmat[ty][tx];
 
}




template<typename T>
__global__ void d_choldc_diagupdate(T *m,
				    int blockoffset, int matsize)  
{
    int boffx = blockIdx.x+blockoffset+1; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

// the +1's stop shared memory bank conflicts when accessing down columns
// There are already no shared bank conflicts when accessing by row

    __shared__ T topt[BLOCK_SIZE][BLOCK_SIZE+1];

// deliberately transposed...
    topt[tx][ty]=MAT_POS( m, matsize, ty+blockoffset*BLOCK_SIZE, tx+boffx*BLOCK_SIZE);

    __syncthreads();

// ty,tx'th thread works out (ty,tx) cmpt of the product...
    T matrixprod=0.;
     

// C'=C-top^T top = C topt topt^T ...
    if(tx>=ty)  
    {

#ifdef UNROLL

	matrixprod+=topt[ty][0]*topt[tx][0];
	matrixprod+=topt[ty][1]*topt[tx][1];
	matrixprod+=topt[ty][2]*topt[tx][2];
	matrixprod+=topt[ty][3]*topt[tx][3];
	matrixprod+=topt[ty][4]*topt[tx][4];
	matrixprod+=topt[ty][5]*topt[tx][5];
	matrixprod+=topt[ty][6]*topt[tx][6];
	matrixprod+=topt[ty][7]*topt[tx][7];
	matrixprod+=topt[ty][8]*topt[tx][8];
	matrixprod+=topt[ty][9]*topt[tx][9];
	matrixprod+=topt[ty][10]*topt[tx][10];
	matrixprod+=topt[ty][11]*topt[tx][11];
	matrixprod+=topt[ty][12]*topt[tx][12];
	matrixprod+=topt[ty][13]*topt[tx][13];
	matrixprod+=topt[ty][14]*topt[tx][14];
	matrixprod+=topt[ty][15]*topt[tx][15];


#else

	for (int kk=0;kk<BLOCK_SIZE;kk++)
	{
	    matrixprod+=topt[ty][kk]*topt[tx][kk];
	}
    
#endif



	MAT_POS( m, matsize, ty+boffx*BLOCK_SIZE, tx+boffx*BLOCK_SIZE )-=matrixprod; 
    }
}






// this kernel takes the results of the above ones and applies them to the 
//rest of the matrix...
template<typename T>
__global__ void d_choldc_hiupdate(T *m,
				  int blockoffset, int matsize, int matblocks)  
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int boffy=blockIdx.x+blockoffset+1;
    int boffx=boffy+1;

// the +1's stop shared memory bank conflicts when accessing down columns
// There are already no shared bank conflicts when accessing by row

    __shared__ T leftt[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ T rightt[BLOCK_SIZE][BLOCK_SIZE+1];


// now read in the data, always from top right

    int tmpx,tmpy,tmpb;

    tmpy=__mul24(boffy,BLOCK_SIZE);
    tmpb=__mul24(blockoffset,BLOCK_SIZE);

// note the tmpy in the latter term to ensure we get the
// correct common matrix for the row
    leftt[tx][ty]=MAT_POS( m, matsize, ty+tmpb, tx+tmpy );

    for (;boffx<matblocks;boffx++){


	tmpx=__mul24(boffx,BLOCK_SIZE);



	rightt[tx][ty]=MAT_POS( m, matsize, ty+tmpb, tx+tmpx );

	__syncthreads();



 
	T matrixprod=0.;

// ty,tx'th thread works out (ty,tx) cmpt of the product...
#ifdef UNROLL


	matrixprod+=leftt[ty][0]*rightt[tx][0];
	matrixprod+=leftt[ty][1]*rightt[tx][1];
	matrixprod+=leftt[ty][2]*rightt[tx][2];
	matrixprod+=leftt[ty][3]*rightt[tx][3];
	matrixprod+=leftt[ty][4]*rightt[tx][4];
	matrixprod+=leftt[ty][5]*rightt[tx][5];
	matrixprod+=leftt[ty][6]*rightt[tx][6];
	matrixprod+=leftt[ty][7]*rightt[tx][7];
	matrixprod+=leftt[ty][8]*rightt[tx][8];
	matrixprod+=leftt[ty][9]*rightt[tx][9];
	matrixprod+=leftt[ty][10]*rightt[tx][10];
	matrixprod+=leftt[ty][11]*rightt[tx][11];
	matrixprod+=leftt[ty][12]*rightt[tx][12];
	matrixprod+=leftt[ty][13]*rightt[tx][13];
	matrixprod+=leftt[ty][14]*rightt[tx][14];
	matrixprod+=leftt[ty][15]*rightt[tx][15];


#else

	for (int kk=0;kk<BLOCK_SIZE;kk++)
	{
	    matrixprod+=leftt[ty][kk]*rightt[tx][kk];
	}
    
#endif

	__syncthreads();

	MAT_POS( m, matsize, ty+tmpy, tx+tmpx )-=matrixprod;

    }

}

template<typename T>
int cuda_potrf(cublasHandle_t handle, char uplo, int n, 
               T *dA, int ldda, int *info)
{
  unsigned MAT_BLOCKS = n/BLOCK_SIZE;
  
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 stripgrid;
  dim3 higrid;

  int j=MAT_BLOCKS;
  int i=j;
  
  if( ldda != n )
  {
     //printf( "ldda <> n\n" );
     return -1;
  }
  
  // MATDISP
  
  while(i>2)
  {
    higrid.x=i-2;
    higrid.y=1;
    higrid.z=1;
    
    
    dim3 stripgrid(i-1);
    
    
    d_choldc_topleft<<<1,threads>>>(dA,j-i,n);
    
    
    d_choldc_strip<<<stripgrid,threads>>>(dA,j-i,n);
    
    d_choldc_diagupdate<<<stripgrid,threads>>>(dA,j-i,n);
    
    /*
    printf("here,%i %i.\n",higrid.x,higrid.y);
    
    error=cudaGetLastError();
    printf("     Error code %d: %s.\n",error,cudaGetErrorString(error));
    */
    
    //      printf("here,%i %i.\n",higrid.x,higrid.y);
    
    d_choldc_hiupdate<<<higrid,threads>>>(dA,j-i,n, MAT_BLOCKS);
    
    /*
    error=cudaGetLastError();
    printf("     Error code %d: %s.\n",error,cudaGetErrorString(error));
    */
    
    i--;
    }
    
    if(j>1)
    {
    
    d_choldc_topleft<<<1,threads>>>(dA,j-2,n);
    
    d_choldc_strip<<<1,threads>>>(dA,j-2,n);
    
    d_choldc_diagupdate<<<1,threads>>>(dA,j-2,n);
  }
  
  d_choldc_topleft<<<1,threads>>>(dA,j-1,n);
  
  return 0;
}

extern "C"{
             
int
cuda_dpotrf(cublasHandle_t handle, char uplo, int n, 
           double *dA, int ldda, int *info)
{
  return cuda_potrf( handle, uplo, n, dA, ldda, info );
}
             
int
cuda_spotrf(cublasHandle_t handle, char uplo, int n, 
           float *dA, int ldda, int *info)
{
  return cuda_potrf( handle, uplo, n, dA, ldda, info );
}

} // extern "C"
