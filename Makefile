CFLAGS ?= -O3 -march=armv8.2-a+simd+fp+fp16fml
CC=gcc-10

all: uk 

uk: uk.o  main.o

#uk_dyn: uk.o uk_dyn.o  main.o

uk.c: blis.py
	exocc -o . --stem $(*F) $^

#uk_dyn.c: blis_dyn.py
#	exocc -o . --stem $(*F) $^

#main.c: uk.c uk_dyn.c
main.c: uk.c 

.PHONY: clean
clean:
	$(RM) uk uk.* *.o 
	$(RM) -r __pycache__/
