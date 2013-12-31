#pragma once
#include <xmmintrin.h>
#include <math.h>

#define abs(x) ((x) >=0 ? (x) : -(x))
#define max(a,b) ((a) > (b)? (a) : (b))
#define min(a,b) ((a) < (b)? (a) : (b))

#define LOGIT(...) do{FILE *f; f = fopen("log.txt","a"); fprintf (f, __VA_ARGS__); fclose(f);}while(0);



typedef struct 
{
    float *eta0s;
    int netas;
    float *lbds;
    int nlambdas;
    int *betas;
    int nbetas;
    float *bias_multipliers;
    int nbias_multipliers;
    int epochs;
    int eval_freq;
    int t;
    float weightPos;
    float weightNeg;
}sgd_cv_params_t;

typedef struct 
{
    float eta0;
    float lbd;
    int beta;
    float bias_multiplier;
    int epochs;
    int eval_freq;
    int t;
    float weightPos;
    float weightNeg;
}sgd_params_t;


typedef struct 
{
    float eta0;
    float lbd;
    int beta;
    float bias_multiplier;
    int t;
    int updates;
    int epoch;
    float acc;
    float weightPos;
    float weightNeg;
}sgd_output_info_t;



struct pq_info_t{
    int nsq;
    int ksq;
    int dsq;
    float *centroids;
    float *centroidsSquaredNorms;
    int nblocks;
};
typedef struct pq_info_t pq_info_t;



void Platts(float *scores, int *labels, int N, float *Aout,float  *Bout);

void vec_addto(float * __restrict__ w,
                      float a, 
                      const float * __restrict__ xi, 
                      long d);

float vec_dotprod(const float * __restrict__ xi,
        const float * __restrict__ w, long d);
int sort(const void *x, const void *y);
void add_slow(int d, float *X, float *A, float f);
void scaleVector_slow(int d, float *W, float f);
float dp_slow(int d, float *a,float *b);


float compute_top1_100(int n, float *scores, int *y);
float compute_top1(int n, float *scores, int *y);

float compute_map_100(int n, float *scores, int *y);
float compute_mapk_100(int n, float *scores, int *y, int k);
float compute_map(int n, float *scores, int *y);
float compute_mapk(int n, float *scores, int *y, int k);


void rpermute(int *a, int n);


void compute_scores(float *W, float B, int n, int d, float *X, float *scores);
void compute_scores_pq(float *W, float B, pq_info_t *pq, int n, int d,  unsigned char *X_pqcodes, float *scores );
float squared_norm2(float *W, int n);

float max_v(float *W,int n);

