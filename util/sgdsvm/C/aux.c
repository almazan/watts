#include <stdio.h>
#include "aux.h"


// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
void Platts(
        float *dec_values, int *labels, int l,
        float * Aout, float * Bout)
{
    float A=0;
    float B=0;
    float prior1=0, prior0 = 0;
    int i;
    
    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;
    
    int max_iter=100;	// Maximal number of iterations
    float min_step=1e-10;	// Minimal step taken in line search
    float sigma=1e-12;	// For numerically strict PD of Hessian
    float eps=1e-5;
    float hiTarget=(prior1+1.0)/(prior1+2.0);
    float loTarget=1/(prior0+2.0);
    float *t=malloc(l*sizeof(float));
    float fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    float newA,newB,newf,d1,d2;
    int iter;
    
    // Initial Point and Initial Fun Value
    A=0.0; B=log((prior0+1.0)/(prior1+1.0));
    float fval = 0.0;
    
    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }
        
        // Stopping Criteria
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;
        
        // Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;
        
        
        stepsize = 1;		// Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;
            
            // New function value
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }
        
        if (stepsize < min_step)
        {
            printf("Line search fails in two-class probability estimates\n");
            break;
        }
    }
    
    if (iter>=max_iter)
        printf("Reaching maximal iterations in two-class probability estimates\n");
    free(t);
    *Aout = A;
    *Bout = B;
}


/* w += xi * a */
void vec_addto(float * __restrict__ w,
        float a,
        const float * __restrict__ xi,
        long d) {
    
    if(((long)xi & 15) == 0 && ((long)w & 15) == 0 && (d & 3) == 0 ) {
        __v4sf *xi4 = (void*)xi;
        __v4sf *w4 = (void*)w;
        __v4sf a4 = {a, a, a, a};
        d /= 4;
#ifdef FASTCACHE
while(d--) *w4 += (*xi4) * a4;
#else
while(d--) *w4 ++ += (*xi4++) * a4;
#endif
return;
    }
    
    /* reference version */
    long j;
    for(j = 0; j < d; j++)
        w[j] += xi[j] * a;
    return;
    
}

float vec_dotprod(const float * __restrict__ xi,
        const float * __restrict__ w, long d) {
    
    if(((long)xi & 15) == 0 && ((long)w & 15) == 0 && (d & 3) == 0 ) {
        __v4sf *xi4 = (void*)xi;
        __v4sf *w4 = (void*)w;
        __v4sf accu4 = {0, 0, 0, 0};
        d /= 4;
#ifndef FASTCACHE
while(d--) accu4 += (*xi4++) * (*w4++);
#else
while(d--) accu4 += (*xi4) * (*w4);
#endif

float *accu = (void*)&accu4;

return accu[0] + accu[1] + accu[2] + accu[3];
    }
    /* reference version */
    long j;
    double accu = 0;
    for(j = 0; j < d; j++)
        accu += xi[j] * w[j];
    return accu;
}

int sort(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

void add_slow(int d, float *X, float *A, float f)
{
    int i;
    for (i=0; i < d; i++)
        X[i]+=f*A[i];
}

void scaleVector_slow(int d, float *W, float f)
{
    int i;
    
    for (i=0; i < d; i++)
        W[i]*=f;
}

float dp_slow(int d, float *a,float *b)
{
    float s=0;
    int i;
    for (i=0; i < d; i++)
        s+= *a++ * *b++;
    return s;
}


float compute_top1_100(int n, float *scores, int *y)
{
    /* Compute top1 in batches of 100 and average */
    float macc = 0;
    for (int i=0; i < n; i+=100)
        macc += compute_top1(100, &scores[i], &y[i]);
    return macc/(n/100.0);

}

float compute_top1(int n, float *scores, int *y)
{
    float max_score = scores[0];
    int idx = 0;    
    /* Check if the first scored sample is correct or not */    
    for (int i=1; i < n; i++)
    {
        if (scores[i] > max_score)
        {
            max_score = scores[i];
            idx = i;
        }
    }
    return y[idx]==1;
}

float compute_mapk_100(int n, float *scores, int *y, int k)
{
/* Compute map@k in batches of 100 and average */
    float mmap = 0;
    for (int i=0; i < n; i+=100)
        mmap += compute_mapk(100, &scores[i], &y[i],k);
    return mmap/(n/100.0);
}


float compute_map_100(int n, float *scores, int *y)
{
    return compute_mapk_100(n, scores, y, n);
}

float compute_map(int n, float *scores, int *y)
{
    return compute_mapk(n, scores, y, n);
}

float compute_mapk(int n, float *scores, int *y, int k) {
    if (k > n) k=n;
    int i,j;
    float prec_at_k;
    int *rank = malloc(n*sizeof(int));
    
    float ave_prec=0;
    int neq = 0;
    int numrel=0;
    float s_score, tmp_score;
    for(i = 0; i < n; i++) {
        if (y[i]!=1)
        {
            continue;
        }
        s_score= scores[i];
        neq = 0;
        rank[numrel]=0;
        // Find how many items (from the class or not)
        for (j=0; j < n; j++){
            if (j==i)
            {continue;}
            tmp_score= scores[j];
            if (tmp_score >  s_score)
            {
                rank[numrel]++;
            }
            else if (tmp_score == s_score)
            {
                neq++;
            }
        }
        rank[numrel]+=floor(neq/2.0);
        numrel++;
    }
    if (numrel==0){free(rank);return 0;}
     
    qsort(rank, numrel, sizeof(int), sort);
   
    /* since we do map@k, numrel equals min(numrel, k) */
    numrel = min(numrel, k);

    for(i=0;i<numrel;i++){
        /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
        if (rank[i] >= k) break;
        prec_at_k =  ((float)(i+1))/(rank[i]+1);
        //mexPrintf("prec_at_k: %f\n", prec_at_k);
        ave_prec += prec_at_k;
    }
    ave_prec/=numrel;
    
    free(rank);
    
    return ave_prec;
}

void rpermute(int *a, int n) {
    
    int k;
    for (k = 0; k < n; k++)
        a[k] = k;
    for (k = n-1; k > 0; k--) {
        int j = rand() % (k+1);
        int temp = a[j];
        a[j] = a[k];
        a[k] = temp;
    }
}

void compute_scores(float *W, float B, int n, int d,  float *X, float *scores )
{
    int i;
    
    for ( i=0; i < n; i++)
    {
        scores[i] = vec_dotprod( W, X+i*d,d);
        scores[i]+=B;
        /*scores[i] = dp_slow(d, W, X+i*d);*/
    }
}

float squared_norm2(float *W, int n)
{
    float norm=0;
    int i;
    for (i=0;i < n;++i) norm+= W[i]*W[i];
    return norm;
}

void compute_scores_pq(float *W, float B,pq_info_t *pq, int n, int d,  unsigned char *X_pqcodes, float *scores )
{
    int i;
    float s;
    float *Wp;
    float *centroidp;
    int q;
    unsigned char *xi;
    float decompressedNorm = 0;
    int dblock = pq->nsq/pq->nblocks;
    for ( i=0; i < n; i++)
    {
        s=0;
        Wp = W;
        centroidp = pq->centroids;
        xi = X_pqcodes + i*pq->nsq;
        decompressedNorm=0;
        float sTmp=0;
        for (q=0; q < pq->nsq; q++)
        {
            sTmp+=vec_dotprod(Wp, centroidp + xi[q]*pq->dsq, pq->dsq);
            //decompressedNorm+=squared_norm2(centroidp + xi[q]*pq->dsq, pq->dsq);
            decompressedNorm+= pq->centroidsSquaredNorms[q*pq->ksq+xi[q]];                                   
            Wp+=pq->dsq;
            centroidp+= pq->dsq*pq->ksq;
            if ((q+1)%dblock==0)
            {
                s += sTmp / sqrt(decompressedNorm);

               decompressedNorm=0;
              sTmp=0; 
            }
        }
        scores[i]=s;
        scores[i]+=B;
        /*scores[i] = dp_slow(d, W, X+i*d);*/
    }
}

float max_v(float *W,int n)
{
    int i;
    float v=W[0];
    for (i=1; i < n; i++) if (W[i] > v) v = W[i];
    return v;
}



