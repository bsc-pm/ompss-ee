#include <stdio.h>
#include <math.h>

const int MAX_NUM_THREADS= 128;

void Particle_array_calculate_forces_opencl(Particle* this_particle_array, Particle *output_array, int number_of_particles, float time_interval ) {
        const int bs = number_of_particles;
        int i;

        for ( i = 0; i < number_of_particles; i += bs )
        {   
		   //Calling the kernel
		   ....(bs,time_interval,number_of_particles,this_particle_array, &output_array[i], i, i+bs-1);	
        }
}