#include "../nbody.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma omp target device(opencl) ndrange(1,size,MAX_NUM_THREADS) copy_deps
#pragma omp task in(d_particles[0;number_of_particles]) out([size] out)
__kernel void calculate_force_func(int size, float time_interval,  int number_of_particles, 
                                              __global Particle* d_particles,__global Particle *out, 
											  int first_local, int last_local); 
#ifdef __cplusplus
}
#endif
