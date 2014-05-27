#!/bin/bash -ex

DIRNAME=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

if [ "X$BSC_MACHINE" == "Xmn3" ]; then
  # (@BSC) Marenostrum III section
  export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/intel/mkl/include/
elif [ "X$BSC_MACHINE" == "Xnvidia" ]; then
  # (@BSC) Minotauro section
  export MKL_LIB_DIR=/opt/compilers/intel/mkl/lib/intel64/
  export MKL_INC_DIR=/opt/compilers/intel/mkl/include/
else
  # Other Machines (AD-HOC) section
  export MKL_LIB_DIR=
  export MKL_INC_DIR=
fi

# Configure OmpSs + Extrae + Paraver
export OMPSS_HOME=/apps/PM/ompss/2014-04-10
export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/latest/default/64
export PARAVER_HOME=/apps/CEPBATOOLS/wxparaver/latest

# Setting environment variables 
export PATH=$OMPSS_HOME/bin:$PATH
export PATH=$EXTRAE_HOME/bin/:$PATH
export PATH=$PARAVER_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH

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
   if [ -s $DIRNAME/common-files/sched-job ] ; then
      echo Not using any job schedule feature for this machine
   else
      echo Job schedule configuration file...
      cat  $DIRNAME/common-files/sched-job
   fi
fi

echo Aditional libraries...

if [ ! -f $MKL_LIB_DIR/libmkl_sequential.so ]; then
   echo \ \ WARNING: MKL library is not found, will not be compiled!
else
   echo \ \ MKL library at $MKL_LIB_DIR
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

