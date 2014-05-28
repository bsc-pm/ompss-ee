PROGRAM=matmul-p

export NX_THREADS=1
export NX_OPENCL_MAX_DEVICES=2 #max number of opencl devices (GPUs in this case) to use

# Creating input file
touch test.in
echo "8192 8192 8192 3" > test.in

# Executing the program
./$PROGRAM

