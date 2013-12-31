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


void train_epoch(int Ntrain, int d, float *Xtrain, int *Ltrain,  int* perm,
        sgd_params_t *params, int updateEta, int *t,  float *W, float *B);




void sgd_train_class_cv(int cls,	
        int Ntrain,int d,          
        float *Xtrain,
        int *Ltrain,
	int Nval,
        float * Xval,
        int *_Lval,
        sgd_cv_params_t *params_cv,
	float *W, float *B,
        float *PlattsA, float *PlattsB,
        sgd_output_info_t *output);

void sgd_train_class(int cls,
        int Ntrain,int d,          
        float *Xtrain,
        int *Ltrain,
	int Nval,
        float *Xval,
        int *Lval,
        sgd_params_t *params,
	float *W, float *B,
        sgd_output_info_t *output);






