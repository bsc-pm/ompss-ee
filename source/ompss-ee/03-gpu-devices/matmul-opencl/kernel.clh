
// Thread block size
#define BLOCK_SIZE 16
//Mercurium pragmas can't "read" values from #defines, so we "save" the value as integer
__constant int BL_SIZE= BLOCK_SIZE;


#ifdef DP
#define REAL double
#else
#define REAL float
#endif

#ifdef __cplusplus
extern "C"
{
#endif

//Kernel declaration as a task should be here
//Remember, we want to multiply two matrices, (A*B=C) where all of them have size NB*NB

#ifdef __cplusplus
}
#endif
