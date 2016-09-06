#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// N and MIN must be powers of 2
long N;
long MIN_SORT_SIZE;
long MIN_MERGE_SIZE;


#define T int


void basicsort(long n, T data[n]);

void basicmerge(long n, T left[n], T right[n], T result[n*2], long start, long length);

void merge_rec(long n, T left[n], T right[n], T result[n*2], long start, long length) {
	if (length < MIN_MERGE_SIZE*2L) {
		// Base case
		#pragma omp task
		basicmerge(n, left, right, result, start, length);
	} else {
		// Recursive decomposition
		
		merge_rec(n, left, right, result, start, length/2);
		merge_rec(n, left, right, result, start + length/2, length/2);
	}
}


void multisort(long n, T data[n], T tmp[n]) {
	if (n >= MIN_SORT_SIZE*4L) {
		// Recursive decomposition
		multisort(n/4L, &data[0], &tmp[0]);
		multisort(n/4L, &data[n/4L], &tmp[n/4L]);
		multisort(n/4L, &data[n/2L], &tmp[n/2L]);
		multisort(n/4L, &data[3L*n/4L], &tmp[3L*n/4L]);
		#pragma omp taskwait

		merge_rec(n/4L, &data[0], &data[n/4L], &tmp[0], 0, n/2L);
		merge_rec(n/4L, &data[n/2L], &data[3L*n/4L], &tmp[n/2L], 0, n/2L);
		#pragma omp taskwait
		merge_rec(n/2L, &tmp[0], &tmp[n/2L], &data[0], 0, n);
		#pragma omp taskwait
	} else {
		// Base case


		#pragma omp task
		basicsort(n, data);

	}
}

static void initialize(long length, T data[length]) {
	for (long i = 0; i < length; i++) {
		if (i==0) {
			data[i] = rand();
		} else {
			data[i] = (data[i-1] * i * 104723L) % N;
		}
		if (data[i] == 0) data[i] = rand();
	}
}

int check_solution(long length, T data[length]) {
	int success=1;
	for (long i = 0; i < length-1; i++) {
		if (data[i]>data[i+1]){
		 success=0; 
		 break;
		}
	}
	return (success);
}
static void touch(long length, T data[length]) {
	for (long i = 0; i < length; i++) {
		data[i] = 0;
	}
}

int main(int argc, char **argv) {
 	int success;

	if (argc != 4) {
		fprintf(stderr, "Usage: %s <vector size in K> <sort size in K> <merge size in K>\n", argv[0]);
		return 1;
	}

	N = atol(argv[1]) * 1024L;
	MIN_SORT_SIZE = atol(argv[2]) * 1024L;
	MIN_MERGE_SIZE = atol(argv[3]) * 1024L;
	
	T *data = malloc(N*sizeof(T));
	T *tmp = malloc(N*sizeof(T));
	
	FILE *fp;
	if((fp=fopen("multisort-leaf.out", "wb"))==NULL) {
		fprintf(stderr, "Unable to open file\n");
		return EXIT_FAILURE;
	}


	double init_time = omp_get_wtime();
	initialize(N, data);
	touch(N, tmp);
	init_time = omp_get_wtime() - init_time;
	double sort_time = omp_get_wtime();

	#pragma omp parallel
	#pragma omp master
	multisort(N, data, tmp);

	sort_time = omp_get_wtime() - sort_time;

	if(fwrite(data, sizeof(T), N, fp) != N) {
		fprintf(stderr, "Output file not written correctly\n");
	}

#if _CHECK_
	success = check_solution(N, data);
	if (!success) printf ("SORTING FAILURE\n"); 
	else printf ("SORTING SUCCESS\n"); 

#endif

    	fprintf(stdout, "Multisort program\n");
    	fprintf(stdout, "   Initialization time in seconds = %g\n", init_time);
    	fprintf(stdout, "   Multisort time in seconds = %g\n", sort_time);
    	fprintf(stdout, "\n");
	return 0;
}
