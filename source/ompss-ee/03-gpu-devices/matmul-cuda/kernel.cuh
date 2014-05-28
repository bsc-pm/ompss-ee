
// Thread block size
#define BLOCK_SIZE 16

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
