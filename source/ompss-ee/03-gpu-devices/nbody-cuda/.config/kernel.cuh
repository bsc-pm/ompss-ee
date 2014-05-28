#include"nbody.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma omp target device(cuda) ndrange(1,size,MAX_NUM_THREADS) copy_deps
#pragma omp task in(d_particles[0;number_of_particles]) out([size] output)
__global__ void calculate_force_func(int size, float time_interval,  int number_of_particles, 
                                              Particle* d_particles, Particle *output, 
											  int first_local, int last_local);
#ifdef __cplusplus
}
#endif
