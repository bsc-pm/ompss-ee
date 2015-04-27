#!/bin/bash -ex

DIRNAME=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

echo Initial configuration...

if [ "X$BSC_MACHINE" == "Xmn3" ]; then
  # (@BSC) Marenostrum III section
  # Configure OmpSs + Extrae + Paraver + Temanejo
  export OMPSS_HOME=/apps/PM/ompss/15.04
  export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/latest/default/64
  export PARAVER_HOME=/apps/CEPBATOOLS/wxparaver/latest
  export TEMANEJO_HOME=
  # Extra package configuration
  export MPI_LIB_DIR=/apps/OPENMPI/1.8.1-mellanox/lib
  export MPI_INC_DIR=/apps/OPENMPI/1.8.1-mellanox/include
  export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/intel/mkl/include/
  export ATLAS_LIB_DIR=
  export ATLAS_INC_DIR=
  ln -sf $DIRNAME/common-files/sched-job_mn3 $DIRNAME/common-files/sched-job
  # Python configuration (needed by Temanejo)
  module load python
elif [ "X$BSC_MACHINE" == "Xnvidia" ]; then
  # (@BSC) Minotauro section
  # Configure OmpSs + Extrae + Paraver + Temanejo
  export OMPSS_HOME=/apps/PM/ompss/15.04
  #export OMPSS_HOME=/gpfs/scratch/bsc56/bsc56678/apps/nvidia/ompss-dev
  export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/latest/default/64
  export PARAVER_HOME=/apps/CEPBATOOLS/wxparaver/latest
  export TEMANEJO_HOME=/apps/PM/ompss/14.09/temanejo
  # Extra package configuration
  export MPI_LIB_DIR=/opt/mpi/bullxmpi/1.1.11.1/lib
  export MPI_INC_DIR=/opt/mpi/bullxmpi/1.1.11.1/include
  export MKL_LIB_DIR=/opt/compilers/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/compilers/intel/mkl/include/
  export ATLAS_LIB_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/lib
  export ATLAS_INC_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/include/
  ln -sf $DIRNAME/common-files/sched-job_minotauro $DIRNAME/common-files/sched-job
  # Python configuration (needed by Temanejo)
  module load python
elif [ "X$BSC_MACHINE" == "XVirtualBox" ]; then
  # (@BSC) VirtualBox section
  # Configure OmpSs + Extrae + Paraver + Temanejo
  export OMPSS_HOME=/home/user/Builds/OmpSs/mcxx
  export EXTRAE_HOME=/home/user/Builds/extrae
  export PARAVER_HOME=/home/user/Tools/paraver
  export TEMANEJO_HOME=/home/user/Builds/temanejo
  # Extra package configuration
  export MPI_LIB_DIR=/usr/lib/openmpi/lib
  export MPI_INC_DIR=/usr/lib/openmpi/include
  export MKL_LIB_DIR=/home/user/Builds/mkl/lib/intel64
  export MKL_INC_DIR=/home/user/Builds/mkl/include
  export ATLAS_LIB_DIR=/usr/lib
  export ATLAS_INC_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/include
  # Python configuration (needed by Temanejo
else
  # Other Machines (AD-HOC) section, fill this section to configure your environment
  # Configure OmpSs + Extrae + Paraver + Temanejo
  export OMPSS_HOME=
  export EXTRAE_HOME=
  export PARAVER_HOME=
  export TEMANEJO_HOME=
  # Extra package configuration
  export MPI_LIB_DIR=
  export MPI_INC_DIR=
  export MKL_LIB_DIR=
  export MKL_INC_DIR=
  export ATLAS_LIB_DIR=
  export ATLAS_INC_DIR=
  touch $DIRNAME/common-files/sched-job
  # Python configuration (needed by Temanejo)
fi

# Setting environment variables 
export PATH=$OMPSS_HOME/bin:$PATH
export PATH=$EXTRAE_HOME/bin/:$PATH
export PATH=$PARAVER_HOME/bin:$PATH
export PATH=$TEMANEJO_HOME/bin:$PATH

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

if [ "X$1" == "X--wipe" ];
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
   popd
fi

