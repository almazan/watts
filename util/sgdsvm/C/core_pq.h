#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>
#pragma once
#include "core.h"
#include "aux.h"


void train_epoch_pq(pq_info_t *pq, int Ntrain, int d, unsigned char * Xtrain_pqcodes, int *Ltrain,  int* perm,
        sgd_params_t *params, int updateEta, int *t,  float *W, float *B);


float evaluateEta_pq(pq_info_t *pq,int Ntrain, int d,  int nmax, unsigned char *Xtrain_pqcodes, int* Ltrain, int *perm, sgd_params_t *params, float eta);
float determineEta0_pq(pq_info_t *pq, int Ntrain, int d,  int nmax,unsigned char *Xtrain_pqcodes,  int* Ltrain,  sgd_params_t *params);


/* With PQ*/
void sgd_train_class_cv_pq(int cls, pq_info_t *pq,	
        int Ntrain,int d,          
        unsigned char *Xtrain_pqcodes,
        int *_Ltrain,
	int Nval,
        unsigned char * Xval_pqcodes,
        int *_Lval,
        sgd_cv_params_t *params_cv,
	float *W, float *B,
        float *PlattsA, float *PlattsB,
        sgd_output_info_t *output);

void sgd_train_class_pq(int cls, pq_info_t *pq,
        int Ntrain,int d,          
        unsigned char *Xtrain_pqcodes,
        int *Ltrain,
	int Nval,
        unsigned char * Xval_pqcodes,
        int *Lval,
        sgd_params_t *params,
	float *W, float *B,
        sgd_output_info_t *output);






