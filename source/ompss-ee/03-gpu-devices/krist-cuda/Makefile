CC=allmcc
NVCC = nvcc

CFLAGS	= --instrument --ompss -O3  --Wn,-w
NVCFLAGS = -O3

all: krist
 

#genera ejecutable
krist:  kernel.o krist.o
	$(CC) $(CFLAGS) -o krist kernel.o krist.o

#compile codigo host (.c)	
krist.o: krist.c krist_auxiliar_header.h clocks.c
	$(CC) $(CFLAGS) -c krist.c

#compile kernel using nvcc (mnvcc also works)
kernel.o: kernel.cu krist_auxiliar_header.h
	$(NVCC) -c kernel.cu
	
clean:
	rm -f krist *.o

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
