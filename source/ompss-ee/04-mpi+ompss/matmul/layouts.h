#ifndef __CUDACC__

#define REAL double

//------------------------------------------------------------
//    Basic data movement tasks
//------------------------------------------------------------

//#pragma omp target device (smp) copy_deps
//#pragma omp task inout ([n]matrix)
void clear_double( int n, REAL (*matrix)[n]);


//#pragma omp target device (smp) copy_deps
#pragma omp task input([N]Alin) output([ts]A) 
void gather_block(int N, REAL (*Alin)[N], int I, int J, int ts, REAL (*A)[ts]);

//alternative declaration. reported bug if copy functions are needed
//In both cases same obejct is passed. 2D matrix of tsxts or NxN
//#pragma omp task input([ts]A) inout([N]Alin) //concurrent ([N]Alin)
//#pragma omp task input(A[0:ts-1]) inout(Alin[0:N-1]) concurrent (Alin[0:N-1])
//#pragma omp task input(A[0:ts-1]) concurrent(Alin[0:N-1]) 
//
//#pragma omp task input(A[0:ts-1]) inout(Alin[0:N-1])
//
//#pragma omp task input(A[0:ts-1]) concurrent(Alin[0:N-1]) 
//#pragma omp target device (smp) copy_deps
//#pragma omp task input(A[0:ts-1]) inout(Alin[0:N-1])
void scatter_block (int ts, REAL (*A)[ts], int N, REAL (*Alin)[N], int I, int J);


//------------------------------------------------------------
//   Memory management and changes in data storage
//------------------------------------------------------------


REAL * malloc_block (int ts);
REAL **  alloc_block_matrix (int nt, int ts);
REAL **  alloc_empty_block_matrix (int nt);

void check_gather_block(int N, REAL (*Alin)[N], int I, int J, int ts, REAL **Ah );


void convert_to_blocks(int N, REAL (*Alin)[N], int DIM, REAL * (*Ah)[DIM]);
void convert_3_to_blocks(int N, REAL (*Alin)[N], REAL (*Blin)[N], REAL (*Clin)[N], 
                         int DIM, REAL * (*Ah)[DIM],  REAL * (*Bh)[DIM],  REAL * (*Ch)[DIM]);

void convert_to_linear(int ts, int DIM, REAL * (*A)[DIM], int N, REAL Alin[N][N]);

void free_block_matrix ( int nt, REAL *A[nt][nt]);
 
#endif // __CUDACC__
