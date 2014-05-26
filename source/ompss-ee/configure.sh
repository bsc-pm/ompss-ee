#!/bin/bash

DIRNAME=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

# Other Machines
#export MKL_LIB_DIR=
#export MKL_INC_DIR=

# Marenostrum (@BSC)
#export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64/
#export MKL_INC_DIR=/opt/intel/mkl/include/

# Minotauro (@BSC)
export MKL_LIB_DIR=/opt/compilers/intel/mkl/lib/intel64/
export MKL_INC_DIR=/opt/compilers/intel/mkl/include/

# Configure OmpSs
export OMPSS_HOME=/apps/PM/ompss/2014-04-10/
export PATH=$OMPSS_HOME/bin:$PATH

# Configure Extrae
export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/latest/default/64/
export PATH=$EXTRAE_HOME/bin/:$PATH

# Configure Paraver
export PARAVER_HOME=/apps/CEPBATOOLS/wxparaver/latest/
export PATH=$PARAVER_HOME/bin:$PATH

# Setting libraries
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH

# Checking configuration
if [ ! -f $DIRNAME/common-files/sched-job ];
then
   echo WARNING: Job schedule file is not configured for this machine!
fi

if [ ! -f $MKL_LIB_DIR/libmkl_sequential.so ];
then
   echo WARNING: MKL library is not found, will not be compiled!
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

