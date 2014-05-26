Introduction
============

This documentation contains examples and exercises using the OmpSs programming model. All the
examples used in this document can be downloaded from the site: http://pm.bsc.es

Getting Started
---------------

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

Job Scheduler Examples
----------------------

The following sections have examples on how to use job scheduler systems installed at BSC machines.

Minotauro
^^^^^^^^^

Slurm is the utility used at MinoTauro for batch processing support, so all jobs must be run through
it. This document provides information for getting started with job execution at the Cluster. These
are the basic directives to submit jobs:

  * mnsubmit <job_script> submits a ''job script'' to the queue system (see below for job script
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

