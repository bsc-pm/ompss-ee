  /* nbody.h */

#ifndef nbody_h
#define nbody_h
 
#define gravitational_constant   6.6726e-11 /* N(m/kg)2 */

typedef struct {
    float position_x; /* m   */
    float position_y; /* m   */
    float position_z; /* m   */
    float velocity_x; /* m/s */
    float velocity_y; /* m/s */
    float velocity_z; /* m/s */
    float mass;       /* kg  */
    float pad;
} Particle;

int MAX_NUM_THREADS= 128;

#endif /* #ifndef nbody_h */

