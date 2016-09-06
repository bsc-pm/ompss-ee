#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// N and MIN must be powers of 2
long N;
long MIN_SORT_SIZE;
long MIN_MERGE_SIZE;

#define T int

int qsort_helper(const void *a, const void *b)
{
	T *realA = (T *)a;
	T *realB = (T *)b;
	return *realA - *realB;
}

void basicsort(long n, T data[n])
{
	qsort(data, n, sizeof(T), qsort_helper);
}

static inline int pivots_are_aligned(T *left, T *right, long n, long leftStart, long rightStart)
{
	if (leftStart == 0 || rightStart == 0 || leftStart == n || rightStart == n) {
		return 1;
	}
	
	if (left[leftStart] <= right[rightStart] && right[rightStart-1] <= left[leftStart]) {
		return 1;
	}
	if (right[rightStart] <= left[leftStart] && left[leftStart-1] <= right[rightStart]) {
		return 1;
	}
	
	return 0;
}

static inline int must_decrease_left(T *left, T *right, long n, long leftStart, long rightStart)
{
	return (left[leftStart] > right[rightStart]);
}

static inline long min(long a, long b)
{
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

static inline void find_pivot(T *left, T *right, long n, long start, long *leftStart, long *rightStart) {
	*leftStart = start/2L;
	*rightStart = start/2L;
	
	if (start == 0) {
		return;
	}
	
	int jumpSize;
	if (pivots_are_aligned(left, right, n, *leftStart, *rightStart)) {
		return;
	} else if (must_decrease_left(left, right, n, *leftStart, *rightStart)) {
		jumpSize = min(start/2L, n - start/2L) / 2L;
		*leftStart -= jumpSize;
		*rightStart += jumpSize;
	} else {
		jumpSize = min(start/2L, n - start/2L) / 2L;
		*leftStart += jumpSize;
		*rightStart -= jumpSize;
	}
	
	while (1) {
		if (pivots_are_aligned(left, right, n, *leftStart, *rightStart)) {
			return;
		} else if (must_decrease_left(left, right, n, *leftStart, *rightStart)) {
			jumpSize = (jumpSize+1L)/2L; // At least jump by 1
			*leftStart -= jumpSize;
			*rightStart += jumpSize;
		} else {
			jumpSize = (jumpSize+1L)/2L; // At least jump by 1
			*leftStart += jumpSize;
			*rightStart -= jumpSize;
		}
	}
}

void seq_merge(long n, T left[n], T right[n], T result[n*2], long start, long length)
{
	long leftStart, rightStart;

	find_pivot(left, right, n, start, &leftStart, &rightStart);
	//printf("Merge %x[%i:%i] %x[%i:%i] -> %x[%i:%i]\n\n", left, leftStart, n, right, rightStart, n, result, start, length);
	
	result += start;
	while (length != 0) {
		if (leftStart == n) {
			*result = right[rightStart];
			rightStart++;
			result++;
		} else if (rightStart == n) {
			*result = left[leftStart];
			leftStart++;
			result++;
		} else if (left[leftStart] <= right[rightStart]) {
			*result = left[leftStart];
			leftStart++;
			result++;
		} else {
			*result = right[rightStart];
			rightStart++;
			result++;
		}
		length--;
	}
}

void merge(long n, T left[n], T right[n], T result[n*2], long start, long length)
{
	if (length < MIN_MERGE_SIZE*2L) {
		// Base case (sequential)
		seq_merge(n, left, right, result, start, length);
	} else {
		// Recursive case (decomposition)
		#pragma omp task 
		merge(n, left, right, result, start, length/2);

		#pragma omp task 
		merge(n, left, right, result, start + length/2, length/2);

		#pragma omp taskwait
	}
}

void multisort(long n, T data[n], T tmp[n])
{
	if (n >= MIN_SORT_SIZE*4L) {
		// Recursive decomposition
		#pragma omp task 
		multisort(n/4L, &data[0], &tmp[0]);
		#pragma omp task 
		multisort(n/4L, &data[n/4L], &tmp[n/4L]);
		#pragma omp task 
		multisort(n/4L, &data[n/2L], &tmp[n/2L]);
		#pragma omp task 
		multisort(n/4L, &data[3L*n/4L], &tmp[3L*n/4L]);

		#pragma omp taskwait

		#pragma omp task 
		merge(n/4L, &data[0], &data[n/4L], &tmp[0], 0, n/2L);
		#pragma omp task 
		merge(n/4L, &data[n/2L], &data[3L*n/4L], &tmp[n/2L], 0, n/2L);
		
		#pragma omp taskwait

		#pragma omp task 
		merge(n/2L, &tmp[0], &tmp[n/2L], &data[0], 0, n);
		#pragma omp taskwait
	} else {
		// Base case
		basicsort(n, data);
	}
}

static void initialize(long length, T data[length])
{
	for (long i = 0; i < length; i++) {
		if (i==0) {
			data[i] = rand();
		} else {
			data[i] = (data[i-1] * i * 104723L) % N;
		}
	}
}

int check_solution(long length, T data[length])
{
	int success=1;
	for (long i = 0; i < length-1; i++) {
		if (data[i]>data[i+1]){
		 success = 0; 
		 break;
		}
	}
	return (success);
}

static void touch(long length, T data[length])
{
	for (long i = 0; i < length; i++) {
		data[i] = 0;
	}
}

int main(int argc, char **argv) {
 	int success;

	if (argc != 4) {
		fprintf(stderr, "Usage: %s <vector size in K> <seq sort size in K> <seq merge size in K>\n", argv[0]);
		return 1;
	}

	N = atol(argv[1]) * 1024L;
	MIN_SORT_SIZE = atol(argv[2]) * 1024L;
	MIN_MERGE_SIZE = atol(argv[3]) * 1024L;
	
	T *data = malloc(N*sizeof(T));
	T *tmp = malloc(N*sizeof(T));
	
	double init_time = omp_get_wtime();
	initialize(N, data);
	touch(N, tmp);
	init_time = omp_get_wtime() - init_time;

	double sort_time = omp_get_wtime();
	multisort(N, data, tmp);
   sort_time = omp_get_wtime() - sort_time;

	success = check_solution(N, data);
	if (!success) printf ("SORTING FAILURE\n"); 
	else printf ("SORTING SUCCESS\n"); 

   fprintf(stdout, "Multisort program (using %d threads)\n", omp_get_num_threads() );
   fprintf(stdout, "   Initialization time in seconds = %g\n", init_time);
   fprintf(stdout, "   Multisort time in seconds = %g\n", sort_time);
   fprintf(stdout, "\n");

	return 0;
}
