MCC=allmcc

CFLAGS=--ompss -O3

OBJECTS = saxpy.o

#link everything 
saxpy: $(OBJECTS)
	${MCC} ${CFLAGS} -o saxpy $(OBJECTS)

#compile saxpy
saxpy.o: saxpy.c
	${MCC} ${CFLAGS} -c saxpy.c kernel.cl 

clean:
	rm -f *.o saxpy

wipe: clean
	$(RM) *.out *.err
	$(RM) *.prv *.row *.pcf
	$(RM) graph.dot graph.pdf
