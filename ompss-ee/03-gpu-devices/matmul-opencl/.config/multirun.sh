PROGRAM=matmul-p

export NX_THREADS=1
export NX_OPENCL_MAX_DEVICES=2 #max number of opencl devices (GPUs in this case) to use
export NX_OPENCL_DEVICE_TYPE=GPU

# Creating input file
touch test.in
echo "4096 4096 4096 3" > test.in

# Executing the program
./$PROGRAM

