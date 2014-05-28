NBody kernel
------------

In this exercise we will work with the NBody kernel. This algorithm numerically approximates the
evolution of a system of bodies in which each body continuously interacts with every other body.
In this case we want to port a traditional SMP source code to another one which can exploit the
beneficts of CUDA/OpenCL. Someone already ported the kernel, so it does the same calculations
than the previous SMP function.

Sources are not complete, we only have the C code which is calling a SMP function and a CUDA/OCL kernel, they do not interact with each other. 

Kernel.c::
  //Call the kernel
  calculate_particles_smp(bs,time_interval,number_of_particles,this_particle_array, &output_array[i], i, i+bs-1);   

In this case there is nothing but a kernel ported by someone and a code calling a smp function.
We’ll need to declare the kernel as an OmpSs task as we have done in previous samples

..note::
Use an intermediate header file and include it, it will work if we declare it on the .c file).

Once the kernel is correctly declared as a task, we can call it instead of the 'old' smp function.
We can proceed to compile the source code, using the command 'make'. After it (if there are no
errors), we can execute it using one of the running scripts. In order to check results, you can
use the command 'diff nbody_out-ref.xyz nbody_out.xyz'.

..note::
If someone is interested, you can try to do a NBody implementation which works with multiple GPUs
can be done if you have finished early, you must split the output in different parts so each GPU
will calculate one of this parts.

If we check the whole source code in nbody.c (not needed), you can see that the
'Particle_array_calculate_forces_cuda' function in kernel.c is called 10 times, and in each call,
the input and output array are swapped, so they act like their counter-part in the next call. So
when we split the output, we must also split the input in as many pieces as the previous output.

In this exercise we will work with the NBody kernel. This algorithm numerically approximates the
evolution of a system of bodies in which each body continuously interacts with every other body

In this case we want to port a traditional SMP source code to another one which can exploit the
beneficts of CUDA/OpenCL. Someone already ported the kernel, so it does the same calculations than
the previous SMP function.

Sources are not complete, we only have the C code which is calling a SMP function and a CUDA/OCL
kernel, they do not interact with each other. 

