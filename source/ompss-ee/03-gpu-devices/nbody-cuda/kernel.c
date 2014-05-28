#include <stdio.h>
#include <math.h>
#include "kernel_header.cuh"

const int MAX_NUM_THREADS= 128;

void Particle_array_calculate_forces_cuda(Particle* this_particle_array, Particle *output_array, int number_of_particles, float time_interval ) {
        const int bs = number_of_particles/1;
        int i;

        for ( i = 0; i < number_of_particles; i += bs )
        {   
		   calculate_force_func(bs,time_interval,number_of_particles,this_particle_array, &output_array[i], i, i+bs-1);	
        }
}
