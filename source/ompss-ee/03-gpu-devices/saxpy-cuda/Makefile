MCC=allmcc
NVCC = mnvcc

NVFLAGS = --Wn,-arch=sm_13
CFLAGS=--ompss -O3 --instrument -lz

OBJECTS = kernel.o saxpy.o

#link everything 
saxpy: $(OBJECTS)
	${MCC} ${CFLAGS} -o saxpy $(OBJECTS)

#compile kernels
kernel.o: kernel.cu kernel_header.h
	${NVCC} ${NVFLAGS} ${CFLAGS} -c kernel.cu

#compile kernels
saxpy.o: saxpy.c
	${MCC} ${CFLAGS} -c saxpy.c

clean:
	rm -f *.o saxpy

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
