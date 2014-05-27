CC=allmcc
CFLAGS	= --instrument --ompss -O3


all: krist

#genera ejecutable
krist:  krist.o
	$(CC) $(CFLAGS) -o krist krist.o

#compile codigo host (.c) & codigo opencl
krist.o: krist.c krist_auxiliar_header.h clocks.c kernel.cl
	$(CC) $(CFLAGS) -c krist.c kernel.cl

clean:
	rm -f krist *.o

run:
	NX_PES=4 NX_ARGS="--gpus=4 --disable-opencl" ./krist 10000 20000

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
