#!/bin/bash -ex

DIRNAME=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

if [ "X$BSC_MACHINE" == "Xmn3" ]; then
  # (@BSC) Marenostrum III section
  export MPI_LIB_DIR=
  export MPI_INC_DIR=
  export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/intel/mkl/include/
  export ATLAS_LIB_DIR=
  export ATLAS_INC_DIR=
  ln -sf $DIRNAME/common-files/sched-job_mn3 $DIRNAME/common-files/sched-job
elif [ "X$BSC_MACHINE" == "Xnvidia" ]; then
  # (@BSC) Minotauro section
  export MPI_LIB_DIR=/opt/mpi/bullxmpi/1.1.11.1/lib
  export MPI_INC_DIR=/opt/mpi/bullxmpi/1.1.11.1/include
  export MKL_LIB_DIR=/opt/compilers/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/compilers/intel/mkl/include/
  export ATLAS_LIB_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/lib
  export ATLAS_INC_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/include/
  ln -sf $DIRNAME/common-files/sched-job_minotauro $DIRNAME/common-files/sched-job
else
  # Other Machines (AD-HOC) section
  export MPI_LIB_DIR=
  export MPI_INC_DIR=
  export MKL_LIB_DIR=
  export MKL_INC_DIR=
  export ATLAS_LIB_DIR=
  export ATLAS_INC_DIR=
  touch $DIRNAME/common-files/sched-job
fi

# Configure OmpSs + Extrae + Paraver
export OMPSS_HOME=/apps/PM/ompss/git
export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/latest/default/64
export PARAVER_HOME=/apps/CEPBATOOLS/wxparaver/latest

# Setting environment variables 
export PATH=$OMPSS_HOME/bin:$PATH
export PATH=$EXTRAE_HOME/bin/:$PATH
export PATH=$PARAVER_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_LIB_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ATLAS_LIB_DIR:$LD_LIBRARY_PATH

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

if [ ! -f $DIRNAME/common-files/sched-job ]; then
   echo WARNING: Job schedule file is not configured for this machine!
else
   echo Job schedule configuration preface...
   cat  $DIRNAME/common-files/sched-job
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

if [ "X$1" == "X--pack" ];
then
   pushd $DIRNAME
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
   pushd ..
   tar -pczf ompss-ee.tar.gz ompss-ee
   popd
   popd
fi

