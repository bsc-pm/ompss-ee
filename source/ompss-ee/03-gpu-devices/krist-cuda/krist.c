#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "krist.h"

long int random(void);
inline int min(int x,int y)
{
    return x<y?x:y;
}
void sincosf(float,float*,float*);
void structfac(int na, int nr, float*a, float*h, float*E);

void structfac_gpuss(int na, int nr, int NA, float*a, int NH, float*h, int NE, float*E);

void printa(int na, float*a);
void printh(int nr,float*h);
void deta(int na, float*a);
void deth(int nr, float*h);
void printhe(int nr, float*h, float*E);
double sumdif(float*a,float*b,int n);

int main(int argc, char*argv[])
{
    int na=1000;   /* number of atoms */
    int nr=10000; /* number of reflections */
    int compute_serial = 0;

    float *h;  /* h[j,0] == h, h[j,1] == k, h[j,2] == l */
    float *E;  /* E[j,0] == real part of E, E[j,1] == imag part of E */
    float *E1;  /* E[j,0] == real part of E, E[j,1] == imag part of E */
    float *a;  /* a[j,0] == atomic number, a[j,1] == x, a[j,2] == y,
                 a[j,3] == z */
    double t0,dt1,dt2;
    int i;

    if (argc > 1) {
        na = atoi(argv[1]);
        nr = atoi(argv[2]);
    }

    if (argc == 4) {
        if (strcmp(argv[3], "--serial") == 0) {
            compute_serial = 1;
        }
    }

    int NH = DIM2_H*nr;
    int NA = DIM2_A*na;
    int NE = DIM2_E*nr;

    printf("Computation of crystallographic normalized structure factors\n"
           "                on the CPU and the GPU\n\n");
    printf("Number of atoms:       %d\n", na);
    printf("Number of reflections: %d\n", nr);

    h = (float*) malloc(sizeof(*h)*DIM2_H*nr);   // 3*10000 30000
    E = (float*) malloc(sizeof(*E)*DIM2_E*nr);   // 2*10000 20000
    E1 = (float*) malloc(sizeof(*E1)*DIM2_E*nr); // 2*10000 20000
    a = (float*) malloc(sizeof(*a)*DIM2_A*na);   // 4*1000   4000

    for (i=0; i<DIM2_E*nr; i++)
        E1[i] = E[i] = 0.0f;

    deta(na,a);
    deth(nr,h);

    if (compute_serial) {
        printf("Running the CPU code\n");
        t0=wallclock();
        structfac(na,nr,a,h,E);
        dt1 = wallclock() - t0;
        //printhe(nr,h,E);
        //printf("Reference: wallclock time seconds:%f\n",dt1);
        printf("computation time (in seconds): %f\n", dt1);
    }

    int times=10000;
    int tt;
    printf("Running the GPU code %d times\n",times);

    t0 = wallclock();
    for (tt=0; tt<times; tt++) {
        structfac_gpuss(na,nr,NA,a,NH,h,NE,E1);
    }
#pragma omp taskwait
    dt2 = (wallclock() - t0) / times ;

    if (compute_serial) {
        printf("Cuda:      wallclock time seconds:%f\n",dt2);
    } else {
        printf("computation time (in seconds): %f\n", dt2);
    }
    double sumdf=sumdif(E,E1,2*nr);
    printf("Cuda:      Sumdif: %f mean: %f\n",sumdf,sumdf/nr);

    return 0;
}

void structfac(int na, int nr, float*a, float*h, float*E)
{
    int i,j;
    float A,B,twopi;
    twopi = 6.28318584f;

    float f2 = 0.0f;
    for (i=0; i<na; i++)
        f2 += a[DIM2_A*i]*a[DIM2_A*i];
    f2 = 1.0f/sqrtf(f2);

    //printf("scaling factor is %f\n",f2);

    for (i=0; i<nr; i++) {
        A=0.0f;
        B=0.0f;
        for (j=0; j<na; j++) {
            float A1,B1;
            float arg = twopi*(h[DIM2_H*i+0]*a[DIM2_A*j+1] +
                               h[DIM2_H*i+1]*a[DIM2_A*j+2] +
                               h[DIM2_H*i+2]*a[DIM2_A*j+3]);
            sincosf(arg, &B1, &A1);
            A += a[DIM2_A*j]*A1;
            B += a[DIM2_A*j]*B1;
        }
        E[DIM2_E*i]   = A*f2;
        E[DIM2_E*i+1] = B*f2;
    }
}


void structfac_gpuss (int na, int nr, int NA, float*a, int NH, float*h, int NE, float*E)
{
    int ii;
    int tasks = 2;

    float f2=0.0f;
    int i;
    for (i=0; i<na; i++)
        f2 += a[i*DIM2_A]*a[i*DIM2_A];
    f2 = 1.0f/sqrtf(f2);

    for (ii = 0; ii < nr; ii += nr/tasks) {
        int nr_2 = nr/tasks;		
		int sharedsize = 16384-2048;
		int maxatoms = sharedsize/(sizeof(float)*DIM2_A);
		//float* shared_mem= (float*) malloc(maxatoms*(sizeof(float)*DIM2_A)); 
        cstructfac(na, nr_2,maxatoms, f2, NA/DIM2_A, a, NH/DIM2_H/tasks, &h[DIM2_H*ii],
                          NE/DIM2_E/tasks, &E[DIM2_E*ii]);
    }
}

void printa(int na, float*a)
{
    int i;
    for (i=0; i<na; i++) {
        printf("atom %d: %f %f %f %f\n",i,a[i*DIM2_A] ,a[i*DIM2_A+1], a[i*DIM2_A+2], a[i*DIM2_A+3]);
    }
}

void printh(int nr,float*h)
{
    int i;
    for (i=0; i<nr; i++) {
        printf("hkl %d: %d %d %d\n",i,(int)h[i*DIM2_H],(int)h[i*DIM2_H+1],(int)h[i*DIM2_H+2]);
    }
}

void deta(int na, float*a)
{
    int i,j;
    for (i=0; i<na; i++) {
        if ( i & 1 )
            a[DIM2_A*i] = 6.0;
        else
            a[DIM2_A*i] = 7.0;
        for (j=1; j<DIM2_A; j++)
            a[DIM2_A*i+j] = (float)random()/(float)RAND_MAX;
    }
}

void deth(int nr, float*h)
{
    const int hmax=20;
    const int kmax=30;
    const int lmax=15;
    int i;
    for (i=0; i<nr; i++) {
        h[DIM2_H*i+0] = rintf(2*hmax*(float)random()/(float)RAND_MAX - hmax);
        h[DIM2_H*i+1] = rintf(2*kmax*(float)random()/(float)RAND_MAX - kmax);
        h[DIM2_H*i+2] = rintf(2*lmax*(float)random()/(float)RAND_MAX - lmax);
    }
}

void printhe(int nr, float*h, float*E)
{
    int i;
    for (i=0; i<nr; i++) {
        printf("hkl %5d: %4d %4d %4d %8g %8g\n",i,(int)h[i*DIM2_H],(int)h[i*DIM2_H+1],
               (int)h[i*DIM2_H+2],E[DIM2_E*i],E[DIM2_E*i+1]);
    }
}


double sumdif(float*a, float*b, int n)
{
    double sum = 0.0;
    int i;
    for (i=0; i<n; i++)
        sum += fabsf(a[i] - b[i]);
    return sum;
}

