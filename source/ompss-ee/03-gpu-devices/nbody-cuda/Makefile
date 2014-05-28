MCC=allmcc
NVCC = mnvcc

NVFLAGS = --Wn,-arch=sm_13
CFLAGS=--ompss -O3 --instrument -lz

OBJECTS = cuda_kernels.o kernel.o nbody.o

#link everything 
nbody: $(OBJECTS)
	${MCC} ${CFLAGS} -o nbody $(OBJECTS)

#compile kernels
cuda_kernels.o: cuda_kernels.cu kernel_header.cuh
	${NVCC} ${NVFLAGS} ${CFLAGS} -c cuda_kernels.cu
#compile kernels
kernel.o: kernel.c kernel_header.cuh
	${MCC} ${CFLAGS} -c kernel.c

#compile kernels
nbody.o: nbody.c nbody.h
	${MCC} ${CFLAGS} -c nbody.c

clean:
	rm -f *.o nbody

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
