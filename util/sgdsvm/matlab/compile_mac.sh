/Applications/MATLAB_R2012b.app/bin/mex  -O CFLAGS="\$CFLAGS -Wall -O3 -ffast-math  -std=c99" -I../C  sgdsvm_train_cv_mex.c  ../C/aux.c ../C/core.c ../C/core_pq.c 
#/Applications/MATLAB_R2012a.app/bin/mex  -g CFLAGS="\$CFLAGS -Wall -g  -std=c99" -I../C  sgdsvm_train_cv_mex.c  ../C/aux.c ../C/core.c ../C/core_pq.c 
