#include <stdio.h>
#include <stdlib.h>

#if _EXTRAE_
#include "extrae_user_events.h"
#endif


#if _EXTRAE_
// Extrae constants
#define PROGRAM		1000
#define END		0
#define SORT		1
#define MERGE		2
#define MULTISORT	3
#define INITIALIZE	4
#endif


// N and MIN_BS must be powers of 2
extern long N;
extern long MIN_SORT_SIZE;
extern long MIN_MERGE_SIZE;

#define T int


int qsort_helper(const void *a, const void *b) {
	//T *realA = (T *)a;
	//T *realB = (T *)b;
	//return *realA - *realB;
	 T realA = *(T *) a;
       T realB = *(T *) b;
       if (realA < realB) return -1;
       else if (realA > realB) return 1;
       return 0;
}


void basicsort(long n, T data[n]) {
#if _EXTRAE_
		Extrae_event(PROGRAM, SORT);
#endif

	//printf("Sort n=%i, %x\n\n", n, data);
	qsort(data, n, sizeof(T), qsort_helper);
#if _EXTRAE_
		Extrae_event(PROGRAM, END);
#endif
}


static inline int pivots_are_aligned(T *left, T *right, long n, long leftStart, long rightStart) {
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


static inline int must_decrease_left(T *left, T *right, long n, long leftStart, long rightStart) {
	return (left[leftStart] > right[rightStart]);
}


static inline long min(long a, long b) {
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


void basicmerge(long n, T left[n], T right[n], T result[n*2], long start, long length) {
	long leftStart, rightStart;

#if _EXTRAE_
		Extrae_event(PROGRAM, MERGE);
#endif

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
#if _EXTRAE_
		Extrae_event(PROGRAM, END);
#endif
}


