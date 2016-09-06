/* nbody.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nbody.h"
#include <time.h>
#include <kernel.h>

#define default_domain_size_x               1.0e+18 /* m  */
#define default_domain_size_y               1.0e+18 /* m  */
#define default_domain_size_z               1.0e+18 /* m  */
#define default_mass_maximum                1.0e+18 /* kg */
#define default_time_interval               1.0e+18 /* s  */
#define default_number_of_particles         1000
#define default_number_of_timesteps         100
#define default_timesteps_between_outputs   default_number_of_timesteps
#define default_random_seed                 12345
#define default_base_filename               "nbody"

#define program_failure_code               -1
#define program_success_code                0

static float mass_maximum;

#define DEFAULT_STRING_LENGTH  1023

static const int    default_string_length = DEFAULT_STRING_LENGTH;
static const int    server = 0;
static char         base_filename[DEFAULT_STRING_LENGTH+1];
static float        domain_size_x;
static float        domain_size_y;
static float        domain_size_z;
static float        time_interval;
static int          number_of_particles;
static int          number_of_timesteps;
static int          timesteps_between_outputs;
static unsigned int random_seed;

static void Particle_input_arguments(FILE *);
static Particle* Particle_array_construct(int number_of_particles);
static void Particle_array_initialize(Particle* this_particle_array, int number_of_particles);
void Particle_array_calculate_forces(Particle* this_particle_array, int number_of_particles);
static void Particle_array_move(Particle* this_particle_array, int number_of_particles, float time_interval);
static void Particle_array_output(char* base_filename, Particle* this_particle_array, int number_of_particles, int timestep);
static Particle* Particle_array_destruct(Particle* this_particle_array, int number_of_particles);
static void Particle_array_output_xyz(FILE * fileptr, Particle* this_particle_array, int number_of_particles);
void Particle_broadcast_arguments(); 

void Particle_array_calculate_forces_cuda(Particle* this_particle_array, Particle *output_array, int number_of_particles, float time_interval ) {
        const int bs = number_of_particles/1;
        int i;

        for ( i = 0; i < number_of_particles; i += bs )
        {   
		   calculate_force_func(bs,time_interval,number_of_particles,this_particle_array, &output_array[i], i, i+bs-1);	
        }
}

double wall_time ( )
{
   struct timespec ts;
   double res;

   clock_gettime(CLOCK_MONOTONIC,&ts);

   res = (double) (ts.tv_sec)  + (double) ts.tv_nsec * 1.0e-9;

   return res;
}

int main (int argc, char** argv)
{ /* main */
    Particle* particle_array = (Particle*)NULL;
    Particle* particle_array2 = (Particle*)NULL;
    int       timestep;
    int       i;
	
   FILE *input_data = fopen(argv[1], "r");
    Particle_input_arguments(input_data);

    particle_array = Particle_array_construct(number_of_particles);
    particle_array2 = Particle_array_construct(number_of_particles);

    Particle_array_initialize(particle_array, number_of_particles);
//    for (i = 0; i < number_of_particles; i++) {
//       particle_array2[i].mass = particle_array[i].mass;
//    }

	FILE * fileptr = fopen("nbody_out.xyz", "w");
	Particle_array_output_xyz(fileptr, particle_array, number_of_particles);

    if (number_of_particles <= 1) return 0;

    double start = wall_time();

    for (timestep = 1; timestep <= number_of_timesteps; timestep++) {

        if ((timestep % timesteps_between_outputs) == 0 ) fprintf(stderr, "Starting timestep #%d.\n", timestep);

        Particle_array_calculate_forces_cuda(particle_array, particle_array2, number_of_particles, time_interval );

       /* swap arrays */
       Particle * tmp = particle_array;
       particle_array = particle_array2;
       particle_array2 = tmp;

    } /* for timestep */

    #pragma omp taskwait
    double end = wall_time ();

      printf("Time in seconds: %g s.\n", end - start);
      printf("Particles per second: %g \n", (number_of_particles*number_of_timesteps)/(end-start));
      
    if ((number_of_timesteps % timesteps_between_outputs) != 0) {
		Particle_array_output_xyz(fileptr, particle_array, number_of_particles);
	}

    particle_array = Particle_array_destruct(particle_array, number_of_particles);
    
    if (fclose(fileptr) != 0) {
		fprintf(stderr, "ERROR: can't close the output file.\n");
        exit(program_failure_code);
	}

    return program_success_code;

} /* main */

/*
 * Get command line arguments.
 */
void Particle_input_arguments (FILE *input)
{ /* Particle_input_arguments */

    number_of_particles       = default_number_of_particles;
    domain_size_x             = default_domain_size_x;
    domain_size_y             = default_domain_size_y;
    domain_size_z             = default_domain_size_z;
    time_interval             = default_time_interval;
    number_of_timesteps       = default_number_of_timesteps;
    timesteps_between_outputs = default_timesteps_between_outputs;
    random_seed               = default_random_seed;
    mass_maximum              = default_mass_maximum;


	    if (fscanf(input, "%d", &number_of_particles) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read number of particles from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%d", &number_of_particles) != 1) */

	    if (number_of_particles < 1) {

	        fprintf(stderr, "ERROR: cannot have %d particles!\n",
	            number_of_particles);

	        exit(program_failure_code);

	    } /* if (number_of_particles < 1) */

	    if (number_of_particles == 1) {

	        fprintf(stderr,
	            "There is only one particle, therefore no forces.\n");

	        exit(program_failure_code);

	    } /* if (number_of_particles == 1) */

	    if (fscanf(input, "%f", &domain_size_x) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read domain size X from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%f", &domain_size_x) != 1) */

	    if (domain_size_x <= 0.0) {

	        fprintf(stderr,
	            "ERROR: cannot have a domain whose X dimension has length %f!\n",
	            domain_size_x);

	        exit(program_failure_code);

	    } /* if (domain_size_x <= 0.0) */

	    if (fscanf(input, "%f", &domain_size_y) != 1) {
               
	        fprintf(stderr,
	            "ERROR: cannot read domain size Y from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%f", &domain_size_y) != 1) */

	    if (domain_size_y <= 0.0) {

	        fprintf(stderr,
	            "ERROR: cannot have a domain whose Y dimension has length %f!\n",
	            domain_size_y);

	        exit(program_failure_code);

	    } /* if (domain_size_y <= 0.0) */

	    if (fscanf(input,"%f", &domain_size_z) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read domain size Z from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%f", &domain_size_z) != 1) */

	    if (domain_size_z <= 0.0) {

	        fprintf(stderr,
	            "ERROR: cannot have a domain whose Z dimension has length %f!\n",
	            domain_size_z);

	        exit(program_failure_code);

	    } /* if (domain_size_z <= 0.0) */

	    if (fscanf(input, "%f", &time_interval) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read time interval from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%f", &time_interval) != 1) */

	    if (time_interval <= 0.0) {

	        fprintf(stderr,
	            "ERROR: cannot have a time interval of %f!\n",
	            time_interval);

	        exit(program_failure_code);

	    } /* if (time_interval <= 0.0) */

	    if (fscanf(input, "%d", &number_of_timesteps) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read number of timesteps from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%d", &number_of_timesteps) != 1) */

	    if (number_of_timesteps <= 0) {

	        fprintf(stderr, "ERROR: cannot have %d timesteps!\n",
	            number_of_timesteps);

	        exit(program_failure_code);

	    } /* if (number_of_timesteps <= 0) */

	    if (fscanf(input, "%d", &timesteps_between_outputs) != 1) {

	        fprintf(stderr,
	        "ERROR: cannot read timesteps between outputs from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%d", &timesteps_between_outputs) != 1) */

	    if (timesteps_between_outputs <= 0) {

	        fprintf(stderr,
	            "ERROR: cannot have %d timesteps between outputs!\n",
	            timesteps_between_outputs);

	        exit(program_failure_code);

	    } /* if (timesteps_between_outputs <= 0) */

	    if (fscanf(input, "%d", &random_seed) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read random seed from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%d", &random_seed) != 1) */

	    if (fscanf(input, "%f", &mass_maximum) != 1) {

	        fprintf(stderr,
	            "ERROR: cannot read mass maximum from standard input!\n");

	        exit(program_failure_code);

	    } /* if (scanf("%f", &mass_maximum) != 1) */

	    if (mass_maximum <= 0.0) {

	        fprintf(stderr,
	            "ERROR: cannot have a maximum mass of %f!\n",
	            mass_maximum);

	        exit(program_failure_code);

	    } /* if (mass_maximum <= 0.0) */

	    fgetc(input);
	    fgets(base_filename, default_string_length, input);
	    if (base_filename[strlen(base_filename)-1] == '\n') {
	        base_filename[strlen(base_filename)-1] = '\0';
	    } /* if (base_filename[strlen(base_filename)-1]...) */

} /* Particle_input_arguments */



#ifdef CHECK
/*
 * Check that the particle exists.
 */
void Particle_check (Particle* this_particle, char* action, char* routine)
{ /* Particle_check */

    if (this_particle != (Particle*)NULL) return;

    fprintf(stderr, "ERROR: can't %s a nonexistent particle\n",
        ((action == (char*)NULL) || (strlen(action) == 0)) ?
            "perform an unknown action on" : action);
    fprintf(stderr, "  in %s\n",
        ((routine == (char*)NULL) || (strlen(routine) == 0)) ?
            "an unknown routine" : routine);

    exit(program_failure_code);

} /* Particle_check */

/*
 * Check that the pair of particles both exist.
 */
void Particle_pair_check (Particle* this_particle1, Particle* this_particle2,
         char* action, char* routine)
{ /* Particle_pair_check */

    if ((this_particle1 != (Particle*)NULL) &&
        (this_particle2 != (Particle*)NULL))
        return;

    fprintf(stderr, "ERROR: can't %s\n",
        ((action == (char*)NULL) || (strlen(action) == 0)) ?
            "perform an unknown action on" : action);

    if (this_particle1 == (Particle*)NULL) {

        if (this_particle2 == (Particle*)NULL) {

            fprintf(stderr, "  a pair of nonexistent particles\n");

        } /* if (this_particle2 == (Particle*)NULL) */
        else {

            fprintf(stderr,
                "  a nonexistent particle and an existing particle\n");

        } /* if (this_particle2 == (Particle*)NULL)...else */

    } /* if (this_particle1 == (Particle*)NULL) */
    else {

        fprintf(stderr,
            "  an existing particle and a nonexistent particle\n");

    } /* if (this_particle1 == (Particle*)NULL)...else */

    fprintf(stderr, "  in %s\n",
        ((routine == (char*)NULL) || (strlen(routine) == 0)) ?
            "an unknown routine" : routine);

    exit(program_failure_code);

} /* Particle_pair_check */
#endif /* #ifdef CHECK */

/*
 * Clear the particle's data.
 */
void Particle_clear (Particle* this_particle)
{ /* Particle_clear */

#ifdef CHECK
    Particle_check(this_particle, "clear", "Particle_clear");
#endif /* #ifdef CHECK */

    this_particle->position_x = 0.0;
    this_particle->position_y = 0.0;
    this_particle->position_z = 0.0;
    this_particle->velocity_x = 0.0;
    this_particle->velocity_y = 0.0;
    this_particle->velocity_z = 0.0;
    this_particle->mass       = 0.0;

} /* Particle_clear */

/*
 * Construct the particle.
 */
void Particle_construct (Particle* this_particle)
{ /* Particle_construct */

#ifdef CHECK
    Particle_check(this_particle, "construct", "Particle_construct");
#endif /* #ifdef CHECK */

    Particle_clear(this_particle);

} /* Particle_construct */

/*
 * Destroy the particle.
 */
void Particle_destruct (Particle* this_particle)
{ /* Particle_destruct */

#ifdef CHECK
    Particle_check(this_particle, "destruct", "Particle_destruct");
#endif /* #ifdef CHECK */

    Particle_clear(this_particle);

} /* Particle_destruct */

/*
 * Initialize the particle by setting its data randomly.
 */
void Particle_set_position_randomly (Particle* this_particle)
{ /* Particle_set_position_randomly */

#ifdef CHECK
    Particle_check(this_particle, "randomly set the position",
        "Particle_set_randomly");
#endif /* #ifdef CHECK */

    this_particle->position_x =
        domain_size_x * ((float) random() / ((float)RAND_MAX + 1.0));
    this_particle->position_y =
        domain_size_y * ((float) random() / ((float)RAND_MAX + 1.0));
    this_particle->position_z =
        domain_size_z * ((float) random() / ((float)RAND_MAX + 1.0));

} /* Particle_set_position_randomly */

/*
 * Initialize the particle by setting its data randomly.
 */
void Particle_initialize_randomly (Particle* this_particle)
{ /* Particle_initialize_randomly */

#ifdef CHECK
    Particle_check(this_particle, "randomly initialize",
        "Particle_initialize_randomly");
#endif /* #ifdef CHECK */

    Particle_clear(this_particle);

    Particle_set_position_randomly(this_particle);

    this_particle->mass       =
        mass_maximum  * ((float) random() / ((float)RAND_MAX + 1.0));

#ifdef CHECK_VAL
printf ("mass %g\n", this_particle->mass);
#endif

} /* Particle_initialize_randomly */

/*
 * Initialize the particle.
 */
void Particle_initialize (Particle* this_particle)
{ /* Particle_initialize */

#ifdef CHECK
    Particle_check(this_particle, "initialize", "Particle_initialize");
#endif /* #ifdef CHECK */

    Particle_initialize_randomly(this_particle);

} /* Particle_initialize */


void Particle_output (FILE* fileptr, Particle* this_particle)
{ /* Particle_output */

    fprintf(fileptr, "%g %g %g %g %g %g %g\n",
        this_particle->position_x,
        this_particle->position_y,
        this_particle->position_z,
        this_particle->velocity_x,
        this_particle->velocity_y,
        this_particle->velocity_z,
        this_particle->mass);

} /* Particle_output */

void Particle_output_xyz (FILE* fileptr, Particle* this_particle) {
	fprintf(fileptr, "C %g %g %g\n",
		this_particle->position_x,
        this_particle->position_y,
		this_particle->position_z);
}

#ifdef CHECK
void Particle_array_check (
         Particle* this_particle_array, int number_of_particles,
         char* action, char* routine)
{ /* Particle_array_check */

    if (number_of_particles < 0) {

        fprintf(stderr, "ERROR: illegal number of particles %d\n",
            number_of_particles);
        fprintf(stderr, "  to %s\n", 
            ((action == (char*)NULL) || (strlen(action) == 0)) ?
                "perform an unknown action on" : action);
        fprintf(stderr, "  in %s\n",
            ((routine == (char*)NULL) || (strlen(routine) == 0)) ?
                "an unknown routine" : routine);

        exit(program_failure_code);

    } /* if (number_of_particles < 0) */

    if (number_of_particles == 0) {

        if (this_particle_array == (Particle*)NULL) return (Particle*)NULL;

        fprintf(stderr, "ERROR: can't %s\n",
            ((action == (char*)NULL) || (strlen(action) == 0)) ?
                "perform an unknown action on" : action);
        fprintf(stderr, "  an existing particle array of length 0\n");
        fprintf(stderr, "  in %s\n",
            ((routine == (char*)NULL) || (strlen(routine) == 0)) ?
                "an unknown routine" : routine);

        exit(program_failure_code);

    } /* if (number_of_particles == 0) */
    if (this_particle_array == (Particle*)NULL) {

        fprintf(stderr, "ERROR: can't %s\n",
            ((action == (char*)NULL) || (strlen(action) == 0)) ?
                "perform an unknown action on" : action);
        fprintf(stderr, "  a nonexistent array of %d particles\n",
            number_of_particles);
        fprintf(stderr, "  in %s\n",
            ((routine == (char*)NULL) || (strlen(routine) == 0)) ?
                "an unknown routine" : routine);

        exit(program_failure_code);

    } /* if (this_particle_array == (Particle*)NULL) */

} /* Particle_array_check */
#endif /* #ifdef CHECK */

/*
 * Allocate and return an array of particles.
 */
Particle* Particle_array_allocate (int number_of_particles)
{ /* Particle_array_allocate */
    Particle* this_particle_array = (Particle*)NULL;

#ifdef CHECK
    if (number_of_particles < 0) {

        fprintf(stderr,
            "ERROR: illegal number of particles %d to allocate\n", 
            number_of_particles);
        fprintf(stderr, "  in Particle_array_construct\n");

        exit(program_failure_code);

    } /* if (number_of_particles < 0) */
#endif /* #ifdef CHECK */

    if (number_of_particles == 0) return (Particle*)NULL;

    this_particle_array =
        (Particle*)malloc(sizeof(Particle) * number_of_particles);
//			posix_memalign( (void**)&this_particle_array, sizeof(Particle) * number_of_particles, sizeof(Particle) * number_of_particles);


    if (this_particle_array == (Particle*)NULL) {

        fprintf(stderr,
            "ERROR: can't allocate a particle array of %d particles\n", 
            number_of_particles);
        fprintf(stderr, "  in Particle_array_construct\n");

        exit(program_failure_code);

    } /* if (this_particle_array == (Particle*)NULL) */

    return this_particle_array;

} /* Particle_array_allocate */

/*
 * Construct and return an array of particles, cleared.
 */
Particle* Particle_array_construct (int number_of_particles)
{ /* Particle_array_construct */
    Particle* this_particle_array = (Particle*)NULL;
    int index;

    this_particle_array =
        Particle_array_allocate(number_of_particles);

    for (index = 0; index < number_of_particles; index++) {

        Particle_construct(&(this_particle_array[index]));

    } /* for index */

    return this_particle_array;

} /* Particle_array_construct */

/*
 * Deallocate the array of particles, and return NULL.
 */
Particle* Particle_array_deallocate (Particle* this_particle_array,
              int number_of_particles)
{ /* Particle_array_deallocate */

#ifdef CHECK
    Particle_array_check(this_particle_array, number_of_particles,
        "deallocate", "Particle_array_deallocate");
#endif /* #ifdef CHECK */

    free(this_particle_array);

    this_particle_array = (Particle*)NULL;

    return (Particle*)NULL;

} /* Particle_array_deallocate */

/*
 * Destroy the array of particles, and return NULL.
 */
Particle* Particle_array_destruct (Particle* this_particle_array,
              int number_of_particles)
{ /* Particle_array_destruct */
    int index;

#ifdef CHECK
    Particle_array_check(this_particle_array, number_of_particles,
        "destroy", "Particle_array_destruct");
#endif /* #ifdef CHECK */

    for (index = number_of_particles - 1; index >= 0; index--) {

        Particle_destruct(&(this_particle_array[index]));

    } /* for index */

    return Particle_array_deallocate(this_particle_array, number_of_particles);

} /* Particle_array_destruct */

/*
 * Initialize the array of particles by setting its data randomly.
 */
void Particle_array_initialize_randomly (
         Particle* this_particle_array, int number_of_particles)
{ /* Particle_array_initialize_randomly */
    int index;

#ifdef CHECK
    Particle_array_check(this_particle_array, number_of_particles,
        "initialize randomly", "Particle_array_initialize_randomly");
#endif /* #ifdef CHECK */

    for (index = 0; index < number_of_particles; index++) {

        Particle_initialize_randomly(&(this_particle_array[index]));

    } /* for index */

} /* Particle_array_initialize_randomly */

/*
 * Initialize the array of particles.
 */
void Particle_array_initialize (
         Particle* this_particle_array, int number_of_particles)
{ /* Particle_array_initialize */

    Particle_array_initialize_randomly(
        this_particle_array, number_of_particles);

} /* Particle_array_initialize */

void Particle_array_output (char* base_filename,
         Particle* this_particle_array, int number_of_particles, int timestep)
{ /* Particle_array_output */
    FILE* fileptr  = (FILE*)NULL;
    char* filename = (char*)NULL;
    int   filename_length;
    int   index;

#ifdef CHECK
    Particle_array_check(this_particle_array, number_of_particles,
        "output", "Particle_array_output");
#endif /* #ifdef CHECK */


    filename_length = strlen(base_filename) + 1 + 8 + 1 + 3;

    filename = (char*)malloc(sizeof(char) * (filename_length + 1));

    if (filename == (char*)NULL) {

        fprintf(stderr, "ERROR: can't allocate the filename string\n");
        fprintf(stderr, "  %s_%8.8d.txt\n", base_filename, timestep);
        fprintf(stderr, "  in Particle_array_output\n");

        exit(program_failure_code);

    } /* if (filename == (char*)NULL) */

    sprintf(filename, "%s_%8.8d.txt", base_filename, timestep);

    fileptr = fopen(filename, "w");

    if (fileptr == (FILE*)NULL) {

        fprintf(stderr, "ERROR: can't open the output file named\n");
        fprintf(stderr, "  %s\n", filename);
        fprintf(stderr, "  in Particle_array_output\n");

        exit(program_failure_code);

    } /* if (fileptr == (FILE*)NULL) */

    fprintf(fileptr, "%d %d %d %g %g %g %g %g %d\n",
        number_of_particles, number_of_timesteps, timesteps_between_outputs,
        domain_size_x, domain_size_y, domain_size_z,
        mass_maximum, time_interval,
        random_seed);

    fprintf(fileptr, "%d\n", timestep);

    for (index = 0; index < number_of_particles; index++) {

        Particle_output(fileptr, &(this_particle_array[index]));

    } /* for index */

    if (fclose(fileptr) != 0) {

        fprintf(stderr, "ERROR: can't close the output file named\n");
        fprintf(stderr, "  %s\n", filename);
        fprintf(stderr, "  in Particle_array_output\n");

        exit(program_failure_code);

    } /* if (fclose(fileptr) != 0) */

    fileptr = (FILE*)NULL;

    free(filename); filename = (char*)NULL;

} /* Particle_array_output */

/* Outputs particle positions in a format that VMD can easily visualize. */
static void Particle_array_output_xyz(FILE * fileptr, 
				Particle* this_particle_array, int number_of_particles) {
	
	fprintf(fileptr, "%d\nNBody\n", number_of_particles);

	int index;
    for (index = 0; index < number_of_particles; index++) {

        Particle_output_xyz(fileptr, &(this_particle_array[index]));

    } /* for index */
} /* Particle_array_output_xyz */
