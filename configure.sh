#!/bin/bash -ex

ROOTNAME=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

if [ "X$1" == "X--pack" ];
then
   pushd $ROOTNAME
   git archive --format=tar.gz --output=ompss-ee.tar.gz --prefix=ompss-ee/ HEAD \
       || { echo >&2 "Option --pack requires git. Aborting"; exit 1; }
   popd
   exit 0
fi

if [ "X$1" == "X--wipe" ];
then
   pushd $ROOTNAME
   for first in `ls -d ??-*/ | cut -f1 -d'/'`;
   do
      echo Entering... $first
      pushd $first
      for second in `ls -d */ | cut -f1 -d'/'`;
      do
         echo Entering... $second
         pushd $second
         make wipe
         popd
      done
      popd
   done
   popd
   exit 0
fi

echo Initial configuration...

if [ "X$BSC_MACHINE" == "X" ]; then
   export BSC_MACHINE=default
fi

export PATH=$ROOTNAME/common-files/:$PATH

source $ROOTNAME/common-files/configure_$BSC_MACHINE

# Setting environment variables 
export PATH=$OMPSS_HOME/bin:$PATH
export PATH=$DLB_HOME/bin:$PATH
export PATH=$EXTRAE_HOME/bin/:$PATH
export PATH=$PARAVER_HOME/bin:$PATH
export PATH=$TEMANEJO_HOME/bin:$PATH
export PATH=$MPI_HOME/bin:$PATH

export LD_LIBRARY_PATH=$MPI_LIB_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ATLAS_LIB_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TEMANEJO_HOME/lib:$LD_LIBRARY_PATH

echo Basic configuration...

# Checking configuration, verbosing configuration
if [ ! -f $OMPSS_HOME/bin/mcc ]; then
   echo \ \ WARNING: Mercurium compiler not found!
else
   echo \ \ Mercurium compiler at $OMPSS_HOME/bin 
fi 

if [ ! -f $EXTRAE_HOME/bin/mpi2prv ]; then
   echo \ \ WARNING: Extrae library not found!
else
   echo \ \ Extrae library at $EXTRAE_HOME/bin 
fi 

if [ ! -f $PARAVER_HOME/bin/wxparaver ]; then
   echo \ \ WARNING: Paraver utility not found!
else
   echo \ \ Paraver utility at $PARAVER_HOME/bin 
fi 

if [ ! -f $TEMANEJO_HOME/bin/Temanejo ]; then
   echo \ \ WARNING: Temanejo utility not found!
else
   echo \ \ Temanejo utility at $TEMANEJO_HOME/bin 
fi 

echo Job schedule SMP configuration preface...
if [ ! -f $ROOTNAME/common-files/sched-job-smp ]; then
   echo \ \ WARNING: Job schedule file for SMP is not configured for this machine!
else
   cat  $ROOTNAME/common-files/sched-job-smp
fi

echo Job schedule MPI configuration preface...
if [ ! -f $ROOTNAME/common-files/sched-job-mpi ]; then
   echo \ \ WARNING: Job schedule file for MPI is not configured for this machine!
else
   cat  $ROOTNAME/common-files/sched-job-mpi
fi

echo Aditional libraries...

if [ ! -f $MPI_LIB_DIR/libmpi.so ]; then
   echo \ \ WARNING: MPI library is not found, some tests will not be compiled!
else
   echo \ \ MPI library at $MPI_LIB_DIR
fi

if [ ! -f $MKL_LIB_DIR/libmkl_sequential.so ]; then
   echo \ \ WARNING: MKL library is not found, some tests will not be compiled!
else
   echo \ \ MKL library at $MKL_LIB_DIR
fi

if [ ! -f $ATLAS_LIB_DIR/libatlas.a ]; then
   echo \ \ WARNING: ATLAS library is not found, some tests will not be compiled!
else
   echo \ \ ATLAS library at $ATLAS_LIB_DIR
fi
