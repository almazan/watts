#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <mex.h>

int sort(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

void mexFunction (int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[]) {
    
    /* Input parameters */
    /* [0]: Similarity matrix
     * [1]: queriesClasses
     * [2]: datasetClasses
     * [3]: NRelevantsPerQuery, number of relevants for each query.
     * [4]: queriesIdx (can be -1s)
     */
    /* Output parameters */
    /* [0]: p@1 array
     * [1]: map array
     * [2]: idx of the best match for each query 
     */
            
    int Nqueries, Ndataset;
    
    int *queriesCls;
    int *datasetCls;
    int *NRelevantsPerQuery;
    int *queriesIdx;        
    float *S;    
          
    
    /* Read Data */
    S =  (float*)mxGetData(prhs[0]);
    Nqueries = (int) mxGetN(prhs[0]);
    Ndataset = (int) mxGetM(prhs[0]);
    queriesCls =  (int*)mxGetData(prhs[1]);    
    datasetCls =  (int*)mxGetData(prhs[2]);
    NRelevantsPerQuery = (int*)mxGetData(prhs[3]);
    queriesIdx =  (int*)mxGetData(prhs[4]);
    
    /* Prepare output */
    mwSize dims[1];
    dims[0]= Nqueries;
    plhs[0] = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    float *pP1 = (float*)mxGetData(plhs[0]);
    float *pMap = (float*)mxGetData(plhs[1]);
    int *bestIdx = (int*)mxGetData(plhs[2]);
    
    
    /* one query per row, scores in each column */
    /* for each query */
    #pragma omp parallel  for
    for (int i=0; i < Nqueries; i++)
    {
        pMap[i]=0;
        pP1[i]=0;
        /* Create a private list of relevants. */
        int *rank = (int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        int Nrelevants = 0;
        /* Get its class */
        int qclass = queriesCls[i];
        /* For each element in the dataset */
        float bestS=-99999;
        int p1=0;
        for (int j=0; j < Ndataset; j++)
        {            
            float s = S[i*Ndataset + j];
            /* Precision at 1 part */
            if (queriesIdx[i]!=j && s > bestS)
            {
                bestS = s;
                p1 = datasetCls[j]==qclass;
                bestIdx[i] = j+1; /* Matlab style */
            }
            /* If it is from the same class and it is not the query idx, it is a relevant one. */
            /* Compute how many on the dataset get a better score and how many get an equal one, excluding itself and the query.*/
            if (datasetCls[j]==qclass && queriesIdx[i]!=j)
            {
                int better=0;
                int equal = 0;
                
                for (int k=0; k < Ndataset; k++)
                {
                    if (k!=j && queriesIdx[i]!=k)
                    {
                        float s2 = S[i*Ndataset + k];
                        if (s2> s) better++;
                        else if (s2==s) equal++;
                    }
                }
                
                
                rank[Nrelevants]=better+floor(equal/2.0);
                Nrelevants++;
            }
        }
        /* Sort the ranked positions) */
        qsort(rank, Nrelevants, sizeof(int), sort);
        
        pP1[i] = p1;
        
        /* Get mAP and store it */
        for(int j=0;j<Nrelevants;j++){
            /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
            
            float prec_at_k =  ((float)(j+1))/(rank[j]+1);
            //mexPrintf("prec_at_k: %f\n", prec_at_k);
            pMap[i] += prec_at_k;            
        }
        pMap[i]/=Nrelevants;
    }
    
    
    
    
    
    
    
    
    return;
    
}

