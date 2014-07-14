
#include <stdlib.h>
#include <mex.h>   /*--This one is required*/

#define max(a,b) ((a) >=(b)?(a):(b))
#define min(a,b) ((a) <(b)?(a):(b))

int overlaps(int i, int j, int *I, int *X, int cols, double overlap)
{
    int x1_i, x2_i, y1_i,y2_i;
    int x1_j, x2_j, y1_j,y2_j;
    int xx_1, xx_2,yy_1,yy_2;
    int w, h;
    double area_i, area_j, ov;
    int r_i,r_j;
    r_i = I[i] -1;
    r_j = I[j] - 1;
    if (X[r_i * cols+4] == X[r_j * cols+4])
    {
        x1_i = X[r_i * cols];
        x2_i = X[r_i * cols+1];
        y1_i = X[r_i * cols+2];
        y2_i = X[r_i * cols+3];
        x1_j = X[r_j * cols];
        x2_j = X[r_j * cols+1];
        y1_j = X[r_j * cols+2];
        y2_j = X[r_j * cols+3];
        area_i = (x2_i-x1_i+1) * (y2_i - y1_i +1);
        area_j = (x2_j-x1_j+1) * (y2_j - y1_j +1);
        xx_1 = max(x1_i,x1_j);
        yy_1 = max(y1_i,y1_j);
        xx_2 = min(x2_i,x2_j);
        yy_2 = min(y2_i,y2_j);
        w = xx_2-xx_1+1;
        h = yy_2-yy_1+1;
        if (w > 0 && h >0)
        {
            ov = w*h / area_j;
            if (ov >= overlap)
                return 1;
        }
    }
    return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i,j;
    int *I;
    int N;
    
    double *out;
    double overlap;
    
    int *X;
    int rows,cols;
    
    int *used;
    int *pick;
    int Npick;
    
    I = (int*)mxGetData(prhs[0]);
    N = mxGetM(prhs[0]);
    
    X = (int*)mxGetData(prhs[1]);
    rows = mxGetN(prhs[1]);
    cols = mxGetM(prhs[1]);
    overlap = mxGetScalar(prhs[2]);
    
	/*
     printf("N: %d\n", N);
     printf("rows: %d cols: %d\n", rows, cols);
     printf("Overlap: %f\n", overlap);
     */
    
    used = (int*)malloc(N * sizeof(int));
    pick = (int*)malloc(N * sizeof(int));
    Npick=0;
    for (i=0; i < N; i++)
    {
        used[i]=0;
        pick[i]=-1;
    }
    for (i=N-1; i >=0;i--)
    {
        if (!used[i])
        {
            used[i]=1;
            pick[Npick++] = I[i];
            for (j=0; j < i; j++)
            {
                if (!used[j] && (overlaps(i,j, I, X, cols, overlap)))
                {
                    used[j]=1;
                }
            }
        }
    }
    plhs[0] = mxCreateDoubleMatrix(Npick,1 , mxREAL);
    out = mxGetPr(plhs[0]);
    for (i=0; i < Npick; i++)
        out[i] = pick[i];
    
    free(used);
    free(pick);
    
    return;
}
