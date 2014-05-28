NVCC = allmcc

NVFLAGS = --Wn,-arch=sm_13
#GCC44_FIX = --compiler-bindir gcc41
CFLAGS=--ompss -O3
#CFLAGS=--ompss -O3 -K -g --instrument -lz
CUDADIR = /opt/cuda/4.0

OBJECTS = nbody.o

nbody: $(OBJECTS)
	${NVCC} ${GCC44_FIX} ${CFLAGS} -L$(CUDADIR)/lib64 -lnanox -o nbody $(OBJECTS) ${LIBS} -lrt

nbody.o: nbody.c nbody.h
	${NVCC} ${GCC44_FIX} ${CFLAGS} -c nbody.c opencl_kernels.cl

clean:
	rm -f *.o nbody

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
