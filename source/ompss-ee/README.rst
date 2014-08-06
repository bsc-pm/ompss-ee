Introduction
============

This documentation contains examples and exercises using the OmpSs programming model.
The main objective of this paper is to provide guidance in learning Ompss programming
model and serve as teaching materials in courses and tutorials. To find more complete
applications please visit our BAR (BSC Application Repository) in the URL
http://pm.bsc.es/projects/bar

.. note::
   A PDF version of this document is available in http://pm.bsc.es/ompss-docs/examples/OmpSsExamples.pdf, and
   all the example source codes in http://pm.bsc.es/ompss-docs/examples/ompss-ee.tar.gz

System configuration
---------------------

In this section we describe how to tune your configure script and also how to use it to configure
your environment. If you have a pre-configured package you can skip this section and simply run
the Linux command ``source`` using your configure script::

  $source configure.sh

The configure script is used to set all environment variables you need to properly execute
OmpSs applications. Among other things it contains the PATH where the system will look for
to find mercurium compiler utility, and the MKL installation directory (if available) to
run specific OmpSs applications (e.g. Cholesky kernel).

To configure your system you have to run the Linux command ``source`` using your configure script::

  $source configure.sh

The make utility
----------------

In software development, Make is a utility that automatically builds executable programs and
libraries from source code by reading files called makefiles which specify how to derive the
target program. Though integrated development environments and language-specific compiler
features can also be used to manage a build process, Make remains widely used, especially in
Unix.

Make searches the current directory for the makefile to use, e.g. GNU make searches files in
order for a file named one of GNUmakefile, makefile, Makefile and then runs the specified (or
default) target(s) from (only) that file.

A makefile consists of rules. Each rule begins with a textual dependency line which defines a
target followed by a colon (:) and optionally an enumeration of components (files or other
targets) on which the target depends. The dependency line is arranged so that the target (left
hand of the colon) depends on components (right hand of the colon). It is common to refer to
components as prerequisites of the target::

  target [target...] : [component...]
  [<TAB>] command-1
  [<TAB>] command-2
  ...
  [<TAB>] command-n
  [target

Below is a very simple makefile that by default compiles the program helloworld (first target:
all) using the gcc C compiler (CC) and using “-g” compiler option (CFLAGS). The makefile also
provides a ''clean'' target to remove the generated files if the user desires to start over (by
running make clean)::

   CC = gcc
   CFLAGS = -g
   LDFLAGS =

   all: helloworld

   helloworld: helloworld.o
      # Commands start with TAB not spaces !!!
      $(CC) $(LDFLAGS) -o $@ $^

   helloworld.o: helloworld.c
      $(CC) $(CFLAGS) -c -o $@ $<

   clean:
      rm -f helloworld helloworld.o


Building the examples
---------------------

All the examples and exercises comes with a makefile (Makefile) configured to compile 3 different
versions for each program. Each of the binnary file name created by doing make ends with a suffix
which determines the version:

 * program-p: performance version
 * program-i: instrumented version
 * program-d: debug version

You can actually select which version you want to compile by executing: ''make program-version''
(e.g. in the Cholesky kernel you can compile the performance version executing ''make cholesky-p''.
By default (running make with no parameters) all the versions are compiled.

Appart of building the program's binnaries, the make utility will also build shell scripts to run
the program. Each exercise have two running scripts, one to run a single program execution
(''run-once.sh''), the other will run multiples configurations with respect the number of threads,
data size, etc (''multirun.sh''). Before submitting any job, make sure all environment variables
have the values you expect to.

In some cases the shell script will contain job scheduler variables declared in top of the script
file. A job scheduler script must contain a series of directives to inform the batch system about
the characteristics of the job. These directives appear as comments in the job script file and the
syntax will depend on the job scheduler system used.

With the running scripts it also comes a ''trace.sh'' file, which can be used to include all the
environment variables needed to get an instrumentation trace of the execution. The content of this
file is::

  #!/bin/bash
  export EXTRAE_CONFIG_FILE=extrae.xml
  export NX_INSTRUMENTATION=extrae
  $*

Finally the make utility will generate (if not already present in the directory) other configuration
files as it is the case of ''extrae.xml'' file (used to configure extrae plugin in order to get a
Paraver trace, see ''trace.sh'' file).

Job Scheduler: Minotauro
------------------------

The current section have a short explanation on how to use the job scheduler systems installed at
Minotauro BSC's machine. Slurm is the utility used in this machine for batch processing support,
so all jobs must be run through it. These are the basic directives to submit jobs:

  * mnsubmit job_script submits a ''job script'' to the queue system (see below for job script
    directives).
  * mnq: shows all the submitted jobs.
  * mncancel <job_id> remove the job from the queue system, canceling the execution of the
    processes, if they were still running.

A job must contain a series of directives to inform the batch system about the characteristics of
the job. These directives appear as comments in the job script, with the following syntax::

   # @ directive = value.

The job would be submitted using: ''mnsubmit <job_script>''. While the jobs are queued you can check
their status using the command ''mnq'' (it may take a while to start executing). Once a job has been
executed you will get two files. One for console standard output (with .out extension) and other
for console standard error (with .err extension).

Job Scheduler: Marenostrum
--------------------------

LSF is the utility used at MareNostrum III for batch processing support, so all jobs must be run
through it. This section provides information for getting started with job execution at the Cluster.
These are the basic commands to submit, control and check your jobs:

  * bsub < job_script: submits a ''job script'' passed through standard input (STDIN) to the queue
    system.
  * bjobs: shows all the submitted jobs
  * bkill <job_id>: remove the job from the queue system, canceling the execution of the processes,
    if they were still running.
  * bsc_jobs: shows all the pending or running jobs from your group.

Liablity Disclaimer
-------------------

The information included in this examples and exercises document and the associated examples's
source file package are not guaranteed to be complete and/or error-free at this stage and they
are subject to changes without furter notice. Barcelona Supercomputing Center will not assume any
responsibility for errors or omissions in this document and/or the associated exmample's source
file package. Please send comments, corrections and/or suggestions to pm-tools at bsc.es



