#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <matrix.h>
#include <sys/time.h>

#include <mex.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
#define MAX(a,b) ((a) >= (b))?(a):(b)
 
int levenshtein(char *s1, char *s2,int s1len,int s2len) {
    unsigned int x, y, lastdiag, olddiag;
    
    unsigned int column[s1len+1];
    for (y = 1; y <= s1len; y++)
        column[y] = y;
    for (x = 1; x <= s2len; x++) {
        column[0] = x;
        for (y = 1, lastdiag = x-1; y <= s1len; y++) {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y-1] + 1, lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return(column[s1len]);
}

void mexFunction (int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[]) {
    
    /* Input parameters */
    /*
     * [0]: w1
     * [1]: w2     
     */
     
    /* Output parameters */
    /* [0]: dist     
     */
            
    int N1, N2;
    char *w1;
    char *w2;    
      
          
    
    /* Read Data */
    w1 = mxArrayToString(prhs[0]);
    w2 = mxArrayToString(prhs[1]);
    N1 = strlen(w1);
    N2 = strlen(w2);
    
    
    /* Prepare output */
    mwSize dims[1];
    dims[0]= 1;
    plhs[0] = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);    
    float *d = (float*)mxGetData(plhs[0]);
    
    *d = levenshtein(w1,w2,N1,N2)/(float)(MAX(N1,N2));                     
    
    return;
    
}


