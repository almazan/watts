#include <mex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define abs(x) ((x) >=0 ? (x) : -(x))

void mexFunction(int nout, mxArray *out[], int nin, const mxArray *in[])
{
    /* in */
    /* [0]: scores        
     * [1]: labels     
     */
    
    /* out */
    /* [0]: [A,B]             
    
     */
    /* Prob = 1 / (1 + exp(A*s + B) ) */
        
    /* variables */    
    
    double *scores;
    double *labels;
    double *t;
    int N;
    double Npos, Nneg;        
    double *m_out;
    int maxIters;
    double minStep, stepSize;
    double sigma;
    
    double h11, h22, h21, g1, g2,gd;
    double p,q,d1,d2, det, dA,dB;
    
    
    double hiTarget, loTarget;
    double fval, fApB;
    double A,B;
    double newA, newB, newf;
    
    
    int i, it;
    /* Init misc */
    scores = mxGetPr(in[0]);
    labels = mxGetPr(in[1]);        
    N = (int)mxGetN(in[0]);
    
    maxIters=100;
    minStep = 1e-10;
    sigma = 1e-12;
    
    t = malloc(N*sizeof(double));
    
    /* Find Npos and Nneg */
    Npos=Nneg=0;
    for (i=0;i<N;i++){
        if (labels[i]>0)
            Npos++;
        else
            Nneg++;
    }    
    /*printf("N: %d, Nneg: %f, Npos: %f\n", N, Nneg, Npos);*/
    
    /* Find Targets */
    hiTarget = (Npos + 1)/(Npos+2);
    loTarget = 1.0/(Nneg+1);
    /*printf("loTarget: %f, hiTarget: %f\n", loTarget, hiTarget);*/
    
    /* Calculate t */    
    for (i=0;i<N;i++){
        if (labels[i]>0)
            t[i] = hiTarget;
        else
            t[i] = loTarget;
    } 
    
    /* Init coeffs */
    A = 0;
    B = log((Nneg+1.0)/(Npos+1.0));
    fval = 0.0;
    for (i=0;i<N;i++){
        fApB = scores[i]*A+B;
        if (fApB >=0)
            fval += t[i]*fApB+log(1+exp(-fApB));
        else
            fval += (t[i]-1)*fApB+log(1+exp(fApB));
    }
    
    
    /* Start iterations */
    
    for (it=0; it < maxIters; it++)
    {
        /*
         * printf("start it %d\n",it);
        printf("A: %f, B: %f, fval: %f fApB: %f\n", A, B, fval, fApB);
         **/
        
        /*Update Gradient and Hessian (use H’ = H + sigma I)*/
        h11=sigma;
        h22=sigma;
        h21=0;
        g1=0;
        g2=0;
        for (i=0;i<N;i++){            
            fApB=scores[i]*A+B;
            if (fApB >= 0){                
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else{
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2 = p*q;
            h11 += scores[i]*scores[i]*d2;
            h22 += d2;
            h21 += scores[i]*d2;
            d1=t[i]-p;            
            g1 += scores[i]*d1;                        
            g2 += d1;
            
        }        
        if (  (abs(g1) < 1e-5)   &&   (abs(g2) < 1e-5)  )
        { 
            /*Stopping criteria*/                        
            break;
        }
        /*Compute modified Newton directions*/
        det=h11*h22-h21*h21;
        dA=-(h22*g1-h21*g2)/det;
        dB=-(-h21*g1+h11*g2)/det;
        gd=g1*dA+g2*dB;
        stepSize = 1;
        while (stepSize >= minStep){ /*Line search*/
            newA=A+stepSize*dA;
            newB=B+stepSize*dB;
            newf=0.0;
            for (i=0;i<N;i++){
                fApB=scores[i]*newA+newB;
                if (fApB >= 0){
                    newf += t[i]*fApB+log(1+exp(-fApB));
                }
                else{
                    newf += (t[i]-1)*fApB+log(1+exp(fApB));
                }                   
            }
            if (newf<fval+0.0001*stepSize*gd){
                /*printf("Write and out\n");*/
                A=newA;
                B=newB;
                fval=newf;
                /*printf("Out by Sufficient decrease satisfied\n");*/
                break; /* Sufficient decrease satisfied */
            }
            else 
            {
                stepSize /= 2.0;
            }
        }
        if (stepSize < minStep){
            puts("Line search fails");
            break;
        }
    }
    if (it >= maxIters)
        puts("Reaching maximum iterations");    
        
    /* Out */
    /*printf("A: %f, B: %f, fval: %f\n", A, B, fval);*/
    out[0] = mxCreateDoubleMatrix(1,2, mxREAL);
	m_out = mxGetPr(out[0]);
    m_out[0]=A;
    m_out[1]=B;
    
}
