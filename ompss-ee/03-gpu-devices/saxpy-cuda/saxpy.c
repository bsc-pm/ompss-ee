#include <kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024*1024
#define BS  64*1024

int main(int argc, char* argv[])
{
   float a=5, *x, *y;
   int i;

   x = (float *) malloc(N*sizeof(float));
   y = (float *) malloc(N*sizeof(float));

   for (i=0; i<N; ++i){
      x[i]=i;
      y[i]=i+2;
   }

   if ( N % BS != 0 ) {
      printf("Size %d should divide block size %d\n", N, BS);
      return -1;
   }

   for (i=0; i<N; i+=BS )
      saxpy(BS, a, &x[i],  &y[i]);

#pragma omp taskwait

   //Check results	
   for (i=0; i<N; ++i){
      if (y[i]!=a*i+(i+2)){
         printf("Error when checking results, in position %d\n",i);
         return -1;
      }
   }

   printf("Results are correct\n");
   return 0;
}
