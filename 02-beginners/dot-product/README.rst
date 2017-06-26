Dot Product
-----------

.. highlight:: c

The dot product is an algebraic operation that takes two equal-length sequences
of numbers and returns a single number obtained by multiplying corresponding
entries and then summing those products. A common implementation of this
operation is shown below::

  double dot_product(int N, int v1[N], int v2[N]) {
    double result = 0.0;
    for (long i=0; i<N; i++)
      result += v1[i] * v2[i];

    return result;
  }

The example above is interesting from a programming model point of view because
it accumulates the result of each iteration on a single variable called
``result``. As we have already seen in this course, this kind of operation is
called reduction, and it is a very common pattern in scientific and
mathematical applications.

There are several ways to parallelize operations that compute a reduction:

 * Protect the reduction with a lock or atomic clause, so that only one thread
   increments the variable at the same time. Note that locks are expensive.
 * Specify that there is a dependency on the reduction variable, but choose
   carefully, you don't want to serialize the whole execution! In this exercise
   we are incrementing a variable, and the sum operation is commutative. OmpSs
   has a type of dependency called ''commutative'', designed specifically for
   this purpose.
 * Use a vector to store intermediate accumulations. Tasks operate on a given
   position of the vector (the parallelism will be determined by the vector
   length), and when all the tasks are completed, the contents of the vector
   are summed.


Once we have introduced the dot product operation and the different ways of
parallelizing a reduction, let's start this exercise. If you open the
*dot-product.c* file, you will see that the ``dot_product`` function is a bit
more complicated than the previous version.

.. Ternary operator is wrongly colored in C syntax. If newer versions ever fix it, c# can be removed.
.. code-block:: c#

    double result = 0.0;
    long j = 0;
    for (long i=0; i<N; i+=CHUNK_SIZE) {
        actual_size = (N - i >= CHUNK_SIZE) ? CHUNK_SIZE : N - CHUNK_SIZE;
        C[j] = 0;

        #pragma omp task label( dot_prod ) firstprivate( j, i, actual_size )
        {
            for (long ii=0; ii<actual_size; ii++)
                C[j] +=  A[i+ii] * B[i+ii];
        }

        #pragma omp task label( increment ) firstprivate( j )
        result += C[j];

        j++;
    }

Basically we have prepared our code to parallelize it, creating a private
storage for each chunk and splitting the main loop into two different nested
loops to adjust the granularity of our tasks (see ``CHUNK_SIZE`` variable).
Apart from that, we have also annotated the tasks for you, but this parallel
version is not ready, yet.


**Goals of this exercise**

 * Find all the ``#pragma omp`` lines. As you can see, there are tasks, but we
   forgot to specify their dependencies.
 * Tasks are executed asynchronously. Thus, at some point we have to wait for
   them. Where should we do that?
 * There is a task with a label ``dot_prod``. What are the inputs of that task?
   Does it have an output?  What is the size of the inputs and outputs?
   Annotate the input and output dependencies.
 * Below the ``dot_prod`` task, there is another task labeled as ``increment``.
   What does it do? Do you see a difference from the previous? You have to
   write the dependencies of this task again, but this time think if there is
   any other clause (besides in and out) that you can use in order to maximize
   parallelism.
 * Think in other parallelization approaches using other types of dependencies.
 * Check scalability (for different versions), use different runtime options (schedulers,...)
 * Get a task dependency graph and/or paraver trackes (analysis)
