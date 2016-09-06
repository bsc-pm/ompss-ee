Stream Benchmark
----------------

The stream benchmark is part of the HPC Challenge benchmarks (http://icl.cs.utk.edu/hpcc/) and here
we present two versions: one that inserts barriers and another without barriers. The behavior of
version with barriers resembles the OpenMP version, where the different functions (Copy, Scale, ...)
are executed one after another for the whole array while in the version without barriers, functions
that operate on one part of the array are interleaved and the OmpSs runtime keeps the correctness
by means of the detection of data-dependences.

.. note::
  You can dowload this code visiting the url http://pm.bsc.es *OmpSs Examples and Exercises*'s
  (code) link. The Stream benchmark is included inside the  *01-examples*'s directory.
