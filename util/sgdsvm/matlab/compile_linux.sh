/usr/local/bin/mex  -O LDFLAGS="\$LDFLAGS -fopenmp" CFLAGS="\$CFLAGS -Wall -O3 -ffast-math -fopenmp -std=c99" -I../C  sgdsvm_train_cv_mex.c  ../C/aux.c ../C/core.c ../C/core_pq.c 
#/softs/stow/matlab-2012b/bin/mex  -g CFLAGS="\$CFLAGS -Wall -g  -std=c99" -I../C  sgdsvm_train_cv_mex.c  ../C/aux.c ../C/core.c ../C/core_pq.c 
