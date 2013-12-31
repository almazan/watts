#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "core.h"
#include "aux.h"


/* Usual entry points are classification with or without crossvalidation */
/* Train one class with cv */
void sgd_train_class_cv(int cls,
        int Ntrain, int d, 
        float *Xtrain,
        int *_Ltrain,
        int Nval, 
        float * Xval,
        int *_Lval,
        sgd_cv_params_t *params_cv,
        float *W, float *B,
        float *PlattsA, float *PlattsB,
        sgd_output_info_t *output)
{
    float bestMap = 0;
    int h,i,j,k,l;


    int *Ltrain = (int*)malloc(Ntrain*sizeof(int));
    int *Lval = (int*)malloc(Nval*sizeof(int));
    for (i=0; i < Ntrain; i++) Ltrain[i] = _Ltrain[i]==cls?1:-1;
    for (i=0; i < Nval; i++) Lval[i] = _Lval[i]==cls?1:-1;

    int ncombs = params_cv->netas * params_cv->nlambdas * params_cv->nbetas * params_cv->nbias_multipliers;
    sgd_params_t *params = (sgd_params_t*)malloc(ncombs*sizeof(sgd_params_t));
    sgd_output_info_t *outputs = (sgd_output_info_t*)malloc(ncombs*sizeof(sgd_output_info_t));

    /* Prepare a validation structure... */
    l=0;
    for (h=0; h < params_cv->netas;h++){
        for (i=0; i < params_cv->nlambdas; i++){
            for (j=0; j < params_cv->nbetas; j++){
                for (k=0; k < params_cv->nbias_multipliers;k++){
                    params[l].eta0 = params_cv->eta0s[h];
                    params[l].lbd = params_cv->lbds[i];
                    params[l].beta = params_cv->betas[j];
                    params[l].bias_multiplier = params_cv->bias_multipliers[k];
                    params[l].epochs = params_cv->epochs;
                    params[l].eval_freq = params_cv->eval_freq;
                    params[l].t = params_cv->t;
                    params[l].weightPos = params_cv->weightPos;
                    params[l].weightNeg = params_cv->weightNeg;
                    l++;
                }
            }
        }
    }

#pragma omp parallel for private(l)
    for (i=0; i < ncombs; i++)
    {
        float *Wtmp = (float*)malloc(d*sizeof(float));
        for (l=0; l < d; l++) Wtmp[l] = 0;
        float Btmp=0;

        sgd_train_class(cls, 
                Ntrain,  d,
                Xtrain,
                Ltrain,
                Nval,
                Xval,
                Lval,
                &params[i],
                Wtmp, &Btmp,
                &outputs[i]);
#pragma omp critical
        {
            if (outputs[i].acc > bestMap)
            {
                bestMap = outputs[i].acc;
                memcpy(W, Wtmp, d*sizeof(float));
                *B = Btmp*params[i].bias_multiplier;

                if (output!=NULL)
                {
                    output->eta0 = outputs[i].eta0;
                    output->lbd = outputs[i].lbd;
                    output->beta = outputs[i].beta;
                    output->bias_multiplier = outputs[i].bias_multiplier;
                    output->updates = outputs[i].updates;
                    output->t = outputs[i].t;
                    output->epoch = outputs[i].epoch;
                    output->acc = outputs[i].acc;
                    output->weightPos = outputs[i].weightPos;
                    output->weightNeg = outputs[i].weightNeg;
                }

            }
        }

    }
    //printf("Finishing class %d with map %.2f and params [%.6f %d %.4f]\n", cls, bestMap, nextlambda[bestComb],nextbeta[bestComb],nextbias[bestComb]);
    float *scoresVal = (float*)malloc(Nval*sizeof(float));
    compute_scores(W, *B, Nval,  d,  Xval, scoresVal);
    Platts(scoresVal, Lval, Nval, PlattsA,PlattsB);
    free(scoresVal);
    free(Ltrain);
    free(Lval);
    free(params);
    free(outputs);
    return;
}

/* Train one class without cv */
void sgd_train_class(int cls, 
        int Ntrain, int d,
        float *Xtrain,
        int *Ltrain,
        int Nval,
        float *Xval,
        int *Lval,
        sgd_params_t *params,
        float *W, float *B,
        sgd_output_info_t *output)
{
    if (output!=NULL)
    {
        output->eta0 = params->eta0;
        output->lbd=params->lbd;
        output->beta=params->beta;
        output->bias_multiplier=params->bias_multiplier;
        output->t=params->t;
        output->updates = 0;
        output->epoch = 0;
        output->acc = 0;
        output->weightPos = params->weightPos;
        output->weightNeg = params->weightNeg;
    }

    int epoch;
    int *perm = (int*)malloc(Ntrain*sizeof(int));


    int t=params->t;


    float *scoresVal = (float*)malloc(Nval*sizeof(float));
    float bestMap = 0;
    float *bestW = (float*)malloc(d * sizeof(float));
    float bestB=0;



    int noimprov=0;
    for (epoch=0; epoch < params->epochs; epoch++)
    {

        /* Create permutation */
        rpermute(perm, Ntrain);


        /* Train epoch */
        train_epoch(Ntrain, d, Xtrain, Ltrain,perm,  params, 1, &t, W,B);
        

        if (Nval!=0 && (epoch %  params->eval_freq==0))
        {

            /*printf("End of epoch %d/%d. %d updates. Accumulated loss: %.2f\n", epoch, epochs,updates, accLoss);*/
            compute_scores(W,*B*params->bias_multiplier,  Nval,  d,    Xval,scoresVal);
            float map = compute_map(Nval, scoresVal, Lval);
            //printf("validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
            if (map > bestMap)
            {
                //LOGIT("%.3f CLS %d with config eta %f lbd %.8f beta %d bm: %f epoch %d\n", map*100, cls,params->eta0, params->lbd,params->beta, params->bias_multiplier, epoch+1);
                noimprov=0;
                /*
                 * printf("Improved validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
                 * mexEvalString("drawnow");
                 */
                memcpy(bestW, W, d*sizeof(float));
                bestB = B[0];
                bestMap = map;
                if (output!=NULL)
                {
                    output->eta0=params->eta0;                    
                    output->acc = bestMap*100;
                    output->epoch = epoch+1;
                    output->updates = t;                                        
                }
            }
            else
            {
                noimprov++;
            }
            if (noimprov==3)
            {
                /* 5 checks without improving. Bail out*/
                break;
            }
        }
    }

    /* One last time... */


    if (Nval!=0)
    {
        /*printf("End of epoch %d/%d. %d updates. Accumulated loss: %.2f\n", epoch, epochs,updates, accLoss);*/
        compute_scores(W,*B*params->bias_multiplier,  Nval,  d,   Xval,  scoresVal);
        float map = compute_map(Nval, scoresVal, Lval);
        if (map > bestMap)
        {
            /*
             * printf("Improved validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
             * mexEvalString("drawnow");
             */
            //LOGIT("%.3f CLS %d with config eta %f lbd %.8f beta %d bm: %f epoch %d\n", map*100, cls,params->eta0, params->lbd,params->beta, params->bias_multiplier, epoch+1);
            memcpy(bestW, W, d*sizeof(float));
            bestB = B[0];
            bestMap = map;
            if (output!=NULL)
            {
                output->eta0=params->eta0;                
                output->acc = bestMap*100;
                output->epoch = params->epochs;
                output->updates = t;                 
            }
        }
    }
    else
    {
        if (output!=NULL)
        {
            output->eta0=params->eta0;
            output->acc = 0;
            output->epoch = params->epochs;
            output->t = t;
        }
    }

    if (Nval!=0)
    {
        /* copy bestW back*/
        memcpy( W, bestW, d*sizeof(float));
        B[0] = bestB;
    }
    free(perm) ;
    free(scoresVal);
    free(bestW);
    return;
}


void train_epoch( int Ntrain, int d, float* Xtrain, int *Ltrain,  int* perm,
        sgd_params_t *params, int updateEta, int *t,  float *W, float *B)
{
    int i;
    float wDivisor;
    int yi;
    float *xi;
    int npos, nneg;
    float L_ovr;
    int updates;
    float accLoss;
    float s;
    float eta;

    /* Set stuff */
    wDivisor = 1;
    npos = 0; nneg = 0;
    updates=0;
    accLoss=0;
    int first = (*t==0?1:0);


    for (i=0; i < Ntrain; i++)
    {
	/* Get the samples */
	xi = Xtrain + perm[i]*d;
        yi = Ltrain[perm[i]];
	//LOGIT("sample %d\n", i);
	//for (int kk=0; kk < 20; kk++) {LOGIT("%.4f ", xi[kk]);} LOGIT("\n");
	//LOGIT("current W: ");
	//for (int kk=0; kk < 20; kk++) {LOGIT("%.4f ", W[kk]);} LOGIT("\n");
	//LOGIT("current B: %.4f\n", *B); 
	//LOGIT("label: %d\n", yi);
        eta = params->eta0;
        if (updateEta)
        {
            if (!first) eta=eta/10.0;
            eta = eta / (1 + params->lbd * eta * *t);

        }
	//LOGIT("eta: %f\n", eta);
        /* Use sample only if pos or if nneg < npos * beta. Otherwise skip
         * sample altogether. IE, skip if sample is negative and there are
         * too many negatives already*/
        if (yi==-1 &&  npos * params->beta <= nneg)
            continue;


        /* Get score*/
        /*s = dp_slow(d, W,xi) / wDivisor;*/
        s= (vec_dotprod(W, xi,d) + *B*params->bias_multiplier) / wDivisor;
	//LOGIT("score: %f\n", s);
        /* Update the daming factor part*/
        wDivisor = wDivisor / (1 - eta * params->lbd);
        /* If things get out of hand, actually update w and set the divisor
         * back to one */
        if (wDivisor > 1e5)
        {
            scaleVector_slow(d,W,1/wDivisor);
            B[0]/=wDivisor;
            wDivisor = 1;
        }

        /* Get Loss*/
        L_ovr = max(0, 1 - yi *s);
	//LOGIT("loss: %f\n", L_ovr);
        if (L_ovr > 0)
        {
            float reweight = (yi==1?params->weightPos:params->weightNeg);
	    vec_addto(W,reweight*yi*eta*wDivisor, xi, d);
            B[0]+= reweight*params->bias_multiplier*yi*eta*wDivisor;
           updates++;
            accLoss+= L_ovr;
        }
        if (yi==1)
            npos++;
        else
            nneg ++;
        (*t) = *t + 1;
    }

    scaleVector_slow(d,W,1.0/wDivisor);
    B[0]/=wDivisor;
    wDivisor = 1;
}








