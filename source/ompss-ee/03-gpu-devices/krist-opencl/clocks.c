#include <time.h>
#include <sys/times.h>
#include <sys/time.h>
#include <stdlib.h>
#include "unistd.h"
double cputime() /* aix, xlf */
{
    struct tms b;
    clock_t r;
    times( &b);
    r = b.tms_utime + b.tms_stime;
    return ( (double) r/(double) sysconf(_SC_CLK_TCK));
}
double CPUTIME() /* cray  */
{
    return ( cputime());
}
double cputime_() /* g77, gcc */
{
    return ( cputime());
}

double wallclock()
{
    struct timeval toot;
    //struct timezone prut;
    double r;

    //gettimeofday(&toot,&prut);
    gettimeofday(&toot, NULL);
    r=toot.tv_sec+0.000001*(double)toot.tv_usec;
    return(r);
}
double WALLCLOCK()
{
    return (wallclock());
}
double wallclock_()
{
    return wallclock();
}

void fortransleep(int *i)
{
    sleep(*i);
}

void FORTRANSLEEP(int *i)
{
    sleep(*i);
}

void fortransleep_(int *i)
{
    sleep(*i);
}

