#include <kernel.h>

#define N 1024

int main(int argc, char* argv[])
{
   float a=5, x[N], y[N];
   int i;

   for (i=0; i<N; ++i){
      x[i]=i;
      y[i]=i+2;
   }

   saxpy(N, a, x,  y);

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
