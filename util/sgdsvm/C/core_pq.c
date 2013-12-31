#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "core.h"
#include "core_pq.h"
#include "aux.h"


/* Usual entry points are classification with or without crossvalidation */
/* Train one class with cv */
void sgd_train_class_cv_pq(int cls, pq_info_t *pq,
        int Ntrain, int d, 
        unsigned char *Xtrain_pqcodes,
        int *_Ltrain,
        int Nval, 
        unsigned char * Xval_pqcodes,
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

        sgd_train_class_pq(cls,pq, 
                Ntrain,  d,
                Xtrain_pqcodes,
                Ltrain,
                Nval,
                Xval_pqcodes,
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
    compute_scores_pq(W, *B, pq, Nval,  d,  Xval_pqcodes, scoresVal);
    Platts(scoresVal, Lval, Nval, PlattsA,PlattsB);
    free(scoresVal);
    free(Ltrain);
    free(Lval);
    free(params);
    free(outputs);
    return;
}

/* Train one class without cv */
void sgd_train_class_pq(int cls, pq_info_t *pq,
        int Ntrain, int d,
        unsigned char *Xtrain_pqcodes,
        int *Ltrain,
        int Nval,
        unsigned char * Xval_pqcodes,
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

    if (params->eta0==0)
    {
        int nmax = Ntrain > 1000?1000:Ntrain;
        params->eta0 = determineEta0_pq(pq, d, Ntrain, nmax, Xtrain_pqcodes, Ltrain, params);
    }


    int noimprov=0;
    for (epoch=0; epoch < params->epochs; epoch++)
    {

        /* Create permutation */
        rpermute(perm, Ntrain);


        /* Train epoch */
        train_epoch_pq(pq, Ntrain, d, Xtrain_pqcodes, Ltrain,perm,  params, 1, &t, W,B);
        


        if (Nval!=0 && (epoch %  params->eval_freq==0))
        {

            /*printf("End of epoch %d/%d. %d updates. Accumulated loss: %.2f\n", epoch, epochs,updates, accLoss);*/
            compute_scores_pq(W,*B*params->bias_multiplier, pq, Nval,  d,    Xval_pqcodes,scoresVal);
            float map = compute_mapk_100(Nval, scoresVal, Lval,3);
            //printf("validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
            if (map > bestMap)
            {
               LOGIT("%.3f CLS %d with config eta %f lbd %.8f beta %d bm: %f epoch %d\n", map*100, cls,params->eta0, params->lbd,params->beta, params->bias_multiplier, epoch+1);
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
        compute_scores_pq(W,*B*params->bias_multiplier, pq, Nval,  d,   Xval_pqcodes,  scoresVal);
        float map = compute_mapk_100(Nval, scoresVal, Lval,3);

        if (map > bestMap)
        {
            /*
             * printf("Improved validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
             * mexEvalString("drawnow");
             */
            LOGIT("%.3f CLS %d with config eta %f lbd %.8f beta %d bm: %f epoch %d\n", map*100, cls,params->eta0, params->lbd,params->beta, params->bias_multiplier, epoch+1);
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


void train_epoch_pq(pq_info_t *pq, int Ntrain, int d, unsigned char * Xtrain_pqcodes, int *Ltrain,  int* perm,
        sgd_params_t *params, int updateEta, int *t,  float *W, float *B)
{
    int i;
    float wDivisor;
    unsigned char *xi;
    int yi;
    int q;

    int npos, nneg;
    float L_ovr;
    int updates;
    float accLoss;
    float s;
    float *Wp;
    float *centroidp;
    float eta;

    /* Set stuff */
    wDivisor = 1;
    npos = 0; nneg = 0;
    updates=0;
    accLoss=0;
    int first = (*t==0?1:0);

    int dblock = 0;
    float decompressedNormTmp;
    float *decompressedNorms=NULL;
    int *q2b = malloc(pq->nsq*sizeof(int));
    dblock = pq->nsq/pq->nblocks;
    decompressedNorms = malloc(pq->nblocks*sizeof(float));
    for (i=0; i < pq->nsq;i++)
        q2b[i] = (int)floor(i/dblock);

    for (i=0; i < Ntrain; i++)
    {
        /* Get the samples */
        xi = Xtrain_pqcodes + perm[i]*pq->nsq;
        yi = Ltrain[perm[i]];
        eta = params->eta0;
        if (updateEta)
        {
            if (!first) eta=eta/10.0;
            eta = eta / (1 + params->lbd * eta * *t);

        }
        /* Use sample only if pos or if nneg < npos * beta. Otherwise skip
         * sample altogether. IE, skip if sample is negative and there are
         * too many negatives already*/
        if (yi==-1 &&  npos * params->beta <= nneg)
            continue;


        /* Get score*/
        /*s = dp_slow(d, W,xi) / wDivisor;*/
        s = 0;
        Wp = W;
        centroidp = pq->centroids;
        decompressedNormTmp=0;
        float sTmp=0;
        for (q=0; q < pq->nsq; q++)
        {
            sTmp+=vec_dotprod(Wp, centroidp + xi[q]*pq->dsq, pq->dsq);
            decompressedNormTmp+= pq->centroidsSquaredNorms[q*pq->ksq+xi[q]];                                   
            Wp+=pq->dsq;
            centroidp+= pq->dsq*pq->ksq;
            if ( (q+1)%dblock==0)
            {
                decompressedNorms[q2b[q]] = 1/sqrt(decompressedNormTmp);
                s += sTmp * decompressedNorms[q2b[q]]; 
                decompressedNormTmp=0;
                sTmp=0; 
            }
        }
        s= (s + B[0]*params->bias_multiplier) / wDivisor;
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
        if (L_ovr > 0)
        {
            float reweight = (yi==1?params->weightPos:params->weightNeg);
            /*add_slow(d,W,xi,yi*eta*wDivisor);*/
            s = 0;
            Wp = W;
            centroidp = pq->centroids;

            for (q=0; q < pq->nsq; q++)
            {
                decompressedNormTmp = decompressedNorms[q2b[q]];
                vec_addto(Wp,reweight*yi*eta*wDivisor*decompressedNormTmp, centroidp + xi[q]*pq->dsq, pq->dsq);
                Wp+=pq->dsq;
                centroidp+= pq->dsq*pq->ksq;
            }
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











/* Auxiliar stuff */
float evaluateEta_pq(pq_info_t *pq, int Ntrain,int d, int nmax, unsigned char *Xtrain_pqcodes, int* Ltrain, int *perm, sgd_params_t *params, float eta)
{    
    float cost=0;
    float loss = 0;
    float wnorm = 0;
    int i;
    float s;
    unsigned char *xi;
    float yi;

    float *W = (float*)malloc(d * sizeof(float));
    for (i=0; i < d; i++)
        W[i]=0;
    float B=0;
    int q;
    float *Wp;
    float *centroidp;
    sgd_params_t newParams;
    newParams.eta0 = eta;
    newParams.lbd = params->lbd;
    newParams.beta = params->beta;
    newParams.bias_multiplier = params->bias_multiplier;
    newParams.epochs = params->epochs;
    newParams.eval_freq = params->eval_freq;
    newParams.t = params->t;

    int t=0;
    train_epoch_pq(pq,nmax,  d,  Xtrain_pqcodes, Ltrain, perm,  &newParams, 0, &t, W, &B);
    float decompressedNorm=0;
    for ( i=0; i<nmax; i++)
    {
        /* Test...*/
        xi = Xtrain_pqcodes + perm[i]*pq->nsq;
        yi = Ltrain[perm[i]];
        s = 0;
        Wp = W;
        centroidp = pq->centroids;
        decompressedNorm=0;
        for (q=0; q < pq->nsq; q++)
        {
            s+=vec_dotprod(Wp, centroidp + xi[q]*pq->dsq, pq->dsq);
            //decompressedNorm+=squared_norm2(centroidp + xi[q]*pq->dsq, pq->dsq);
            decompressedNorm+= pq->centroidsSquaredNorms[q*pq->ksq+xi[q]];                                   
            Wp+=pq->dsq;
            centroidp+= pq->dsq*pq->ksq;
        }
        s= (s/sqrt(decompressedNorm) + B*params->bias_multiplier);

        /* Get Loss*/
        loss+= max(0, 1 - yi *s);
    }

    for (i=0; i < d; i++) wnorm+=(W[i]*W[i]);
    wnorm = sqrt(wnorm);

    loss = loss / nmax;
    cost = loss + 0.5 * params->lbd * wnorm;
    //printf("Trying eta=%.6f  yields cost %.2f\n",eta, cost);

    return cost;
}


float determineEta0_pq(pq_info_t *pq, int d, int Ntrain, int nmax,unsigned char *Xtrain_pqcodes,  int* Ltrain, sgd_params_t *params)
{
    float eta0;
    float factor = 2.0;
    float eta1 = 0.5;
    float eta2 = eta1 * factor;

    int *perm = (int*)malloc(Ntrain*sizeof(int));
    rpermute(perm, Ntrain);

    float cost1 = evaluateEta_pq(pq, Ntrain,d,   nmax, Xtrain_pqcodes, Ltrain, perm, params, eta1);
    float cost2 = evaluateEta_pq(pq,Ntrain,d,    nmax, Xtrain_pqcodes,  Ltrain, perm, params, eta2);
    if (cost2 > cost1)
    {
        float tmp = eta1; eta1 = eta2; eta2 = tmp; 
        tmp = cost1; cost1 = cost2; cost2 = tmp; 
        factor = 1 / factor;
    }
    do
    {
        eta1 = eta2; 
        eta2 = eta2 * factor;
        cost1 = cost2;  
        cost2 = evaluateEta_pq(pq,Ntrain,d,    nmax, Xtrain_pqcodes,  Ltrain, perm, params, eta2);
    }while (cost1 > cost2);
    eta0 = eta1 < eta2?eta1:eta2;
    if (eta0 > 1)
    {
        eta0=1;
    }
    free(perm);

    return eta0;
}
