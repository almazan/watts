#include <stdio.h>
#include "aux.h"
#include "core_pq.h"

int main(void)
{
    printf("In main!\n");

    /* Prepare sgd params */
    sgd_cv_params_t p;
    p.eta0s = malloc(3*sizeof(float)); p.eta0s[0] = 1.0; p.eta0s[1] = 0.1; p.eta0s[2]=0.01;p.netas = 3;
    p.lbds = malloc(4*sizeof(float)); p.lbds[0] = 1e-3; p.lbds[1] = 1e-4;p.lbds[2] = 1e-5; p.lbds[3] = 1e-6; p.nlambdas = 4;
    //p.lbds = malloc(1*sizeof(float)); p.lbds[0] = 1e-6; p.nlambdas = 1;
    p.betas = malloc(1*sizeof(int)); p.betas[0] = 64;p.nbetas = 1;
    p.bias_multipliers = malloc(1*sizeof(float)); p.bias_multipliers[0] = 1.0;p.nbias_multipliers=1;
    p.epochs = 60;
    p.eval_freq = 3;
    p.t = 0;
    p.weightPos = 2;
    p.weightNeg = 1;

    /* Prepare pq info */
    pq_info_t pq;
    pq.nsq = 5120;
    pq.ksq = 256;
    pq.dsq = 16;
    pq.centroids = malloc(pq.nsq*pq.dsq*pq.ksq*sizeof(float));
    FILE *f = fopen("pqfile.dat","rb");
    fread(pq.centroids, sizeof(float), pq.nsq*pq.dsq*pq.ksq, f);
    fclose(f);
    pq.centroidsSquaredNorms = malloc(pq.nsq*pq.ksq*sizeof(float));
    int cp=0;
    int cps=0;
    for (int i=0; i < pq.nsq;i++){
        for (int j=0; j < pq.ksq;j++){
            for (int k=0; k < pq.dsq;k++){
                pq.centroidsSquaredNorms[cps]+= (pq.centroids[cp]*pq.centroids[cp]);
                cp++;
            }
            cps++;
        }
    }
    pq.nblocks = 10;
    /* Prepare data */
    int Ntrain = 5478;
    int Nval = 2800;
    int d = pq.nsq*pq.dsq;
    unsigned char *Xtrain_pqcodes = malloc(Ntrain*pq.nsq*sizeof(unsigned char));
    f = fopen("newTrainCodes.dat","rb");
    fread(Xtrain_pqcodes, sizeof(unsigned char), Ntrain*pq.nsq,f);
    fclose(f);
    unsigned char *Xval_pqcodes = malloc(Nval*pq.nsq*sizeof(unsigned char));
    f = fopen("newValidCodes.dat","rb");
    fread(Xval_pqcodes, sizeof(unsigned char), Nval*pq.nsq,f);
    fclose(f);
    
    int *Ltrain = malloc(Ntrain*sizeof(int));
    f = fopen("newTrainLabels.dat","rb");
    fread(Ltrain, sizeof(int), Ntrain,f);
    fclose(f);

    int *Lvalid = malloc(Nval*sizeof(int));
    f = fopen("newValidLabels.dat","rb");
    fread(Lvalid, sizeof(int), Nval,f);
    fclose(f);

    /* Prepare class */
    float *W = malloc(d*sizeof(float));
    float B, PlattsA, PlattsB;

    /* Prepare output */
    sgd_output_info_t output;

    /* Run */
    sgd_train_class_cv_pq(0, &pq,
            Ntrain, d,
            Xtrain_pqcodes,
            Ltrain,
            Nval,
            Xval_pqcodes,
            Lvalid,
            &p,
            W, &B,
            &PlattsA,&PlattsB,
            &output);
        
    /*
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
    */

    return 0;
}

