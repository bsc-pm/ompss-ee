Dot Product
-----------

.. highlight:: c

The dot product is an algebraic operation that takes two equal-length sequences of numbers and
returns a single number obtained by multiplying corresponding entries and then summing those
products. This is the code that you can found in dot_product() function::

  for (long i=0; i<N; i+=CHUNK_SIZE) {
    actual_size = (N-CHUNK_SIZE>=CHUNK_SIZE)?CHUNK_SIZE:(N-CHUNK_SIZE);
    C[j]=0;
    for (long ii=0; ii<actual_size; ii++) {
       C[j]+= A[i+ii] * B[i+ii];
    }
    acc += C[j];
    j++;
  }

This simple mathematical operation is performed on 2 vectors of N-dimension and returns an scalar.
It is interesting (at least from the programming model point of view) because you have to accumulate
your result on a single variable (acc), and you want to do this from multiple threads at the same
time. This is called a reduction and, as we have already seen in the course, there are several ways
to do it:

 * Protect the reduction with a lock or atomic clause, so that only one thread increments the
   variable at the same time. Note that locks are expensive.
 * Specify that there is a dependency on this variable, but choose carefully, you don't want to
   serialize the whole execution! In this exercise we are incrementing a variable, and the sum
   operation is commutative. OmpSs has a type of dependency called ''commutative'', designed
   specifically for this purpose.
 * Use a vector to store intermediate accumulations. Tasks operate on a given position of the
   vector (the parallelism will be determined by the vector length), and when all the tasks are
   completed, the contents of the vector are summed.

In that code we have already annotated the tasks for you, but this parallel version is not ready, yet.

 * Find all the #pragma omp lines. As you can see, there are tasks, but we forgot to specify the
   dependencies.
 * There is a task with a label dot_prod. What are the inputs of that task? Does it have an output?
   What is the size of the inputs and outputs? Annotate the input and output dependencies.
 * Below the dot_prod task, there is another task labeled as increment. What does it do? Do you see
   a difference from the previous? You have to write the dependencies of this task again, but this
   time think if there is any other clause (besides in and out) that you can use in order to maximize
   parallelism.
