gcc -Wall -O3 -ffast-math -fopenmp -std=c99 -fPIC -shared -Wl,-soname=libsgdsvm.so -I./C ./C/aux.c ./C/core.c ./C/core_pq.c -o libsgdsvm.so

