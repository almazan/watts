#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <mex.h>

#include "core.h"


mxArray *parseOutput(sgd_output_info_t *output)
{
    /* Very ugly... */
    const char *infonames[10] = {"eta0", "lbd", "beta", "bias_multiplier", "t","updates","epoch","acc", "weightPos", "weightNeg"};
    /* Allocate array for the structure */
    mwSize dims[1];
    dims[0]=1;
    
    mxArray *infoArray = mxCreateStructMatrix(1, 1, 10, infonames);
    /* Allocate arrays for the data */
    mxArray *eta0A = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *lambdaA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *betaA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *bias_multiplierA = mxCreateNumericArray(1,dims, mxSINGLE_CLASS, mxREAL);
    mxArray *tA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *updatesA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *epochA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *accA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *wpA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *wnA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    /* Associate with arrays and write... */
    float *eta0 = (float*)mxGetData(eta0A); eta0[0] = output->eta0;
    float *lbd = (float*)mxGetData(lambdaA); lbd[0] = output->lbd;
    int *beta = (int*)mxGetData(betaA); beta[0] = output->beta;
    float *bias_multiplier = (float*)mxGetData(bias_multiplierA); bias_multiplier[0] = output->bias_multiplier;
    int *t = (int*)mxGetData(tA); t[0] = output->t;
    int *updates = (int*)mxGetData(updatesA); updates[0] = output->updates;
    int *epoch = (int*)mxGetData(epochA); epoch[0] = output->epoch;
    float *acc = (float*)mxGetData(accA); acc[0] = output->acc;
    float *wp = (float*)mxGetData(wpA); wp[0] = output->weightPos;
    float *wn = (float*)mxGetData(wnA); wn[0] = output->weightNeg;
    /* set fields ...*/
    mxSetField(infoArray, 0, infonames[0], eta0A);
    mxSetField(infoArray, 0, infonames[1], lambdaA);
    mxSetField(infoArray, 0, infonames[2], betaA);
    mxSetField(infoArray, 0, infonames[3], bias_multiplierA);
    mxSetField(infoArray, 0, infonames[4], tA);
    mxSetField(infoArray, 0, infonames[5], updatesA);
    mxSetField(infoArray, 0, infonames[6], epochA);
    mxSetField(infoArray, 0, infonames[7], accA);
    mxSetField(infoArray, 0, infonames[8], wpA);
    mxSetField(infoArray, 0, infonames[9], wnA);
    return infoArray;
}


sgd_cv_params_t *parseOpts(const mxArray *paramsArray){
    
    sgd_cv_params_t *params = mxMalloc(sizeof(sgd_cv_params_t));
    const char *fnames[9] = {"eta0s","lbds", "betas", "bias_multipliers","epochs","eval_freq", "t", "weightPos", "weightNeg"};
    mxArray *eta0sA;
    mxArray *lbdsA;
    mxArray *betasA;
    mxArray *bias_multipliersA;
    eta0sA = mxGetField(paramsArray, 0, fnames[0]);
    lbdsA = mxGetField(paramsArray, 0, fnames[1]);
    betasA = mxGetField(paramsArray, 0, fnames[2]);
    bias_multipliersA = mxGetField(paramsArray, 0, fnames[3]);

    if (!mxIsClass(eta0sA, "single"))
    {
        printf("Error in etas. Expected single precision\n");
        return NULL;
    }
     
    if (!mxIsClass(lbdsA, "single"))
    {
        printf("Error in lambdas. Expected single precision\n");
        return NULL;
    }
    
    if (!mxIsClass(betasA, "int32"))
    {
        printf("Error in betas. Expected int32\n");
        return NULL;
    }
    if (!mxIsClass(bias_multipliersA, "single"))
    {
        printf("Error in bias_multipliers. Expected single precision\n");
        return NULL;
    }
    
   
    
    params->eta0s =  (float*)mxGetData(eta0sA);
    params->netas = (int) mxGetN(eta0sA);
    params->lbds =  (float*)mxGetData(lbdsA);
    params->nlambdas = (int) mxGetN(lbdsA);
    params->betas =  (int*)mxGetData(betasA);
    params->nbetas = (int) mxGetN(betasA);
    params->bias_multipliers =  (float*)mxGetData(bias_multipliersA);
    params->nbias_multipliers = (int) mxGetN(bias_multipliersA);
    params->epochs =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[4]));
    params->eval_freq =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[5]));
    params->t =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[6]));
    params->weightPos =(float)mxGetScalar(mxGetField(paramsArray, 0, fnames[7]));
    params->weightNeg =(float)mxGetScalar(mxGetField(paramsArray, 0, fnames[8]));
    
    return params;   
}

void displayParams(sgd_cv_params_t *params)
{
    printf("Displaying cv_params\n");
    printf("eta0s [%d]: ",params->netas); for (int i=0; i < params->netas; i++) printf("%.4f ",params->eta0s[i]); printf("\n");
    printf("lbds [%d]: ",params->nlambdas); for (int i=0; i < params->nlambdas; i++) printf("%.8f ",params->lbds[i]); printf("\n");
    printf("betas [%d]: ",params->nbetas); for (int i=0; i < params->nbetas; i++) printf("%d ",params->betas[i]); printf("\n");
    printf("bias_multipliers [%d]: ",params->nbias_multipliers); for (int i=0; i < params->nbias_multipliers; i++) printf("%.4f ",params->bias_multipliers[i]); printf("\n");
    printf("epochs: %d\n", params->epochs);
    printf("eval_freq: %d\n", params->eval_freq);
    printf("t: %d\n", params->t);
    printf("weightPos: %.2f\n", params->weightPos);
    printf("weightNeg: %.2f\n", params->weightNeg);
    }

void mexFunction (int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[]) {
    
    /* Input parameters */
    /* [0]: Xtrain
     * [1]: Ltrain
     * [2]: Xvalid
     * [3]: Lvalid
     * [4]: opts...
     *
     * [x]: eta0s
     * [x]: lambdas
     * [x]: betas
     * [x]: bias_multipliers
     * [x]: epochs
     * [x]: eval_freq
     * [x]: t
     * [x]: weightPos
     * [x]: weightNeg
     */
    /* Output parameters */
    /* [0]: W,B,PlattsA,PlattsB,info
     *
     */


    int d;
    int Ntrain, Nval;
    float *Xtrain;
    float *Xval;
    int *Ltrain;
    int *Lval;
    sgd_cv_params_t *params;
    const mxArray  *paramsArray;
    
    
    float *W;
    float *B;
    float *PlattsA;
    float *PlattsB;
    
    
    /* Read Data */
    Xtrain =  (float*)mxGetData(prhs[0]);
    d = (int) mxGetM(prhs[0]);
    Ntrain = (int) mxGetN(prhs[0]);
    Ltrain = (int*)mxGetData(prhs[1]);
    Xval =  (float*)mxGetData(prhs[2]);
    Nval = (int) mxGetN(prhs[2]);
    Lval = (int*)mxGetData(prhs[3]);
    paramsArray = prhs[4];
    
    params = parseOpts(paramsArray);        
    /*displayParams(params);*/
    
    if (params==NULL)
        return;
    
    
    /* Allocate output data */
    /*const char *fnames[4] = {"W","B", "PlattsA", "PlattsB"};*/
    const char *fnames[5] = {"W","B", "PlattsA", "PlattsB","info"};
    
    
    plhs[0] = mxCreateStructMatrix(1, 1,5, fnames);
    mwSize dims[1];
    dims[0]=d;
    mxArray *Warray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    dims[0] = 1;
    mxArray *Barray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *PlattsAarray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *PlattsBarray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    W = (float*)mxGetData(Warray);
    B = (float*)mxGetData(Barray);
    PlattsA = (float*)mxGetData(PlattsAarray);
    PlattsB = (float*)mxGetData(PlattsBarray);

    /* Train! */
    /* Since we asked the positive class to be "1", we train class 1... */
    sgd_output_info_t output;
    
    sgd_train_class_cv(1,Ntrain,d,
            Xtrain,Ltrain,
            Nval,
            Xval,Lval,
            params,
            W,B,PlattsA,PlattsB,
            &output);
    
       
    mxArray *infoarray = parseOutput(&output);
    
    
    mxSetField(plhs[0], 0, fnames[0], Warray);
    mxSetField(plhs[0], 0, fnames[1], Barray);
    mxSetField(plhs[0], 0, fnames[2], PlattsAarray);
    mxSetField(plhs[0], 0, fnames[3], PlattsBarray);
    mxSetField(plhs[0], 0, fnames[4], infoarray);
    mxFree(params); 
    return;
    
}

