PROGRAM=matmul-p

export NX_SMP_WORKERS=1
export NX_GPUS=2 #change this in order to use more GPUs

export NX_GPUMAXMEM=90

# Creating the input file
touch test.in
echo "8192 8192 8192 3" > test.in

# Executing the application
./$PROGRAM
