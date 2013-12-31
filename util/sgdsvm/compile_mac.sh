gcc -Wall -O3 -ffast-math  -std=c99 -fPIC -shared -Wl,-dylib_install_name,libsgdsvm.so -I./C ./C/aux.c ./C/core.c ./C/core_pq.c -o libsgdsvm.so

