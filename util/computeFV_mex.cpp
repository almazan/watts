#include "mex.h"
#include "string.h"
#include <string>
#include <cmath>
#include <map>
#include <xmmintrin.h>

extern "C" {
#include <vl/generic.h>
#include <vl/imopv.h>
#include <vl/mathop.h>
#include <vl/dsift.h>
#include <vl/fisher.h>
}

#define INF ((int)999999)

enum {GAUSSIAN, TRIANGULAR} ;



int max_int(int *seq,int N)
{
    int m = seq[0];
    for (int i=1; i < N; i++)
        if (seq[i] > m)
            m=seq[i];
    return m;
}

typedef struct {
    int floatDescriptors;
    int fast;
    int *sizes;
    int nsizes;
    int step;
    float magnif;
    float windowSize;
    int bounds[4];
    int currentSize;
    int norm;
    float contrastthreshold;
}PHOWopts;


typedef struct img_t{
    int H;
    int W;
    float *data;
}img_t;

typedef struct {
    float *eigv;
    float *mu;
    int D;
    int d;
}pca_t;

typedef struct{
    float *w;
    float *mu;
    float *sigma;
    int D;
    int G;
} gmm_t;

typedef struct{
    int x;
    int y;
    int w;
    int h;
    int cx;
    int cy;
}boundingBox_t;


void vec_addto(float * __restrict__ w,
        float a,
        const float * __restrict__ xi,
        long d) {
    
    if(((long)xi & 15) == 0 && ((long)w & 15) == 0 && (d & 3) == 0 ) {
        __v4sf *xi4 = (__v4sf*)(void*)xi;
        __v4sf *w4 = (__v4sf*)(void*)w;
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
        __v4sf *xi4 = (__v4sf*)(void*)xi;
        __v4sf *w4 = (__v4sf*)(void*)w;
        __v4sf accu4 = {0, 0, 0, 0};
        d /= 4;
#ifndef FASTCACHE
while(d--) accu4 += (*xi4++) * (*w4++);
#else
while(d--) accu4 += (*xi4) * (*w4);
#endif

float *accu = (float*)(void*)&accu4;

return accu[0] + accu[1] + accu[2] + accu[3];
    }
    /* reference version */
    long j;
    double accu = 0;
    for(j = 0; j < d; j++)
        accu += xi[j] * w[j];
    return accu;
}


void freePCA(pca_t **model)
{
    if (*model!=NULL)
    {
        if((*model)->eigv!=NULL) free((*model)->eigv);
        if((*model)->mu!=NULL)  free((*model)->mu);
        free(*model);
    }
    *model = NULL;
}

void freeGMM(gmm_t **model)
{
    if (*model!=NULL)
    {
        if((*model)->w!=NULL) free((*model)->w);
        if((*model)->mu!=NULL)  free((*model)->mu);
        if((*model)->sigma!=NULL)  free((*model)->sigma);
        free(*model);
    }
    *model = NULL;
}

void freeImages(img_t ***images, int nImages)
{
    if (*images!=NULL)
    {
        img_t ** p = *images;
        for (int i=0; i < nImages;i++)
        {
            if (p[i]!=NULL)
            {
                if (p[i]->data!=NULL) free(p[i]->data);
                free(p[i]);
            }
            
        }
        free(*images);
    }
    *images = NULL;
}


pca_t * readPCA(char *pcaPath){
    pca_t *model = (pca_t*)malloc(sizeof *model);
    FILE *f = fopen(pcaPath, "rb");
    fread(&(model->d), sizeof(int), 1, f);
    fread(&(model->D), sizeof(int), 1, f);
    model->eigv = (float*)malloc(model->d*model->D*sizeof(float));
    model->mu = (float*)malloc(model->D*sizeof(float));
    fread(model->eigv, sizeof(float), model->d * model->D, f);
    fread(model->mu, sizeof(float), model->D, f);
    return model;
}

gmm_t * readGMM(char *gmmPath){
    gmm_t *model = (gmm_t*)malloc(sizeof *model);
    FILE *f = fopen(gmmPath, "rb");
    fread(&(model->G), sizeof(int), 1, f);
    fread(&(model->D), sizeof(int), 1, f);
    model->w = (float*)malloc(model->G*sizeof(float));
    model->mu = (float*)malloc(model->G*model->D*sizeof(float));
    model->sigma = (float*)malloc(model->G*model->D*sizeof(float));
    fread(model->w, sizeof(float), model->G, f);
    fread(model->mu, sizeof(float), model->G*model->D, f);
    fread(model->sigma, sizeof(float), model->G*model->D, f);
    return model;
}



void writeMat(char *outPath,float *mat, int N, int D )
{
    FILE *f = fopen(outPath, "wb");
    fwrite(&N, sizeof(int), 1, f);
    fwrite(&D, sizeof(int), 1, f);
    fwrite(mat, sizeof(float), N*D, f);
    fclose(f);
}



float otsu(img_t *img)
{
    int size = img->H*img->W;
    float isize = 1.0/size;
    float hist[256] = {0};
    for (int i=0; i < size ; i++)
        hist[(int)round(img->data[i]*255)]++;
    for (int i=0; i < 256; i++)
        hist[i]*=isize;
    // Calculate total mean level
    float mu_t = 0;
    for (int i=0; i<256; i++)
        mu_t+=(i+1)*hist[i];
    
    float *sigma_b_squared = (float*)malloc(255*sizeof(float));
    float maxval = 0;
    float omega = 0;
    float mu=0;
    for (int i=0; i < 256;i++)
    {
        omega += hist[i];
        mu += (i+1)*hist[i];
        float denom = (omega * (1-omega));
        sigma_b_squared[i] = denom < 0.00001?0:((mu_t * omega - mu)*(mu_t * omega - mu))/denom;
        if (sigma_b_squared[i] > maxval) maxval = sigma_b_squared[i];
    }
    
    float acc=0;int c=0;
    for (int i=0; i < 256;i++)
    {
        if (abs(sigma_b_squared[i] - maxval) < 0.000001)
        {
            acc+=i;
            c++;
        }
    }
    return ((acc/c)/255.0);
}


void extractWordBox(img_t *image, float px,float py, boundingBox_t *box){
    /* Get threshold using otsu */
    float grayThresh = otsu(image);

    int rows = image->H;
    int cols = image->W;
    float *countRows = (float*)malloc(rows*sizeof(float));
    float *countCols = (float*)malloc(cols*sizeof(float));
    float *countRowsCum = (float*)malloc((1+rows)*sizeof(float));
    float *countColsCum = (float*)malloc((1+cols)*sizeof(float));
    int count;
    count = 0;
    for (int i=0; i < rows; i++) countRows[i]=0;
    for (int i=0; i < cols; i++) countCols[i]=0;
    /* Count number of content pixels (ie, v < th) for rows, cols, and total */
    float *p = image->data;
    for (int c=0; c < cols; c++)
    {
        for (int r=0; r < rows; r++)
        {
            
            if (*p <= grayThresh)
            {
                countRows[r]++;
                countCols[c]++;
                count++;
            }
            p++;
        }
    }
    
    countRowsCum[0]=0;
    countColsCum[0]=0;
    
    
    /* Create normalised cummulative histograms on the fly */
    for (int i=1; i < rows+1; i++) countRowsCum[i] = countRowsCum[i-1] + countRows[i-1]/count;
    for (int i=1; i < cols+1; i++) countColsCum[i] = countColsCum[i-1] + countCols[i-1]/count;
    

    
    /* Find the shortest sequence that contains more than px or py pixels */
    
    int bestLength = 99999;
    int bestStart = 0;
    int bestEnd = 0;
    
    for (int i=0; i < rows; i++)
        for (int j=i+1; j < rows+1; j++)
        {
        if (j-i +1 > bestLength) break;
        if (countRowsCum[j] - countRowsCum[i] > py && j -i +1 < bestLength)
        {
            bestLength = j-i+1;
            bestStart = i;
            bestEnd = j;
        }
        }
    box->y = bestStart;
    box->h = bestEnd - bestStart + 1;
    box->cy = box->y + box->h/2;
    bestLength = 99999;
    bestStart = 0;
    bestEnd = 0;
    
    for (int i=0; i < cols; i++)
        for (int j=i+1; j < cols+1; j++)
        {
        if (j-i +1 > bestLength) break;
        if (countColsCum[j] - countColsCum[i] > px && j -i +1 < bestLength)
        {
            bestLength = j-i+1;
            bestStart = i;
            bestEnd = j;
        }
        }
    box->x = bestStart;
    box->w = bestEnd - bestStart + 1;
    box->cx = box->x + box->w/2;
    mexPrintf("%d %d %d %d\n", box->x, box->y, box->w, box->h);
    return;
}


float *appendXY(float *descr, float *frames, int N, int D, boundingBox_t *bbox){
    /* Create output, save space for 2 extra dims */
    float *out = (float*)malloc(N*(D+2)*sizeof(float));
    float *pd = descr;
    float *pout = out;
    float *pf = frames;
    for (int n=0; n < N; n++)
    {
        /* Copy projected sift */
        memcpy(pout, pd, D*sizeof(float));
        pd+=D;
        pout+=D;
        /* Copy projected frame */
        *pout++ = (*pf++ - bbox->cx)/bbox->w;
        *pout++ = (*pf++ - bbox->cy)/bbox->h;
        /* skip contrast and scale */
        pf+=2;
    }
    return out;
}

float *ApplyPca(float *data, int N, int Din, int dout, pca_t *model)
{
    
    assert(model->D == Din);
    assert(model->d >= dout);
    float *out = (float *)malloc(N * dout * sizeof(float));
    //for (int i=0; i < N*dout; i++) out[i]=0.0;
    
#define D(y,x) data[(y)*Din + (x)]
#define O(y,x) out[(y)*dout + (x)]
#define E(y,x) model->eigv[(y)*Din + (x)]
#define M(x) model->mu[(x)]


for (int n=0; n < N; n++)
{
    vec_addto(&data[n*Din], -1, model->mu, Din);
    //for (int D=0; D < Din; D++)
    //    D(n,D)-=M(D);
}


for (int n=0; n < N; n++)
{
    for (int d=0; d < dout;d++ )
    {
        O(n,d) = vec_dotprod(&data[n*Din], &model->eigv[d*Din], Din);
        //O(n,d)=0.0;
        //for (int D=0; D < Din; D++)
        //{
        //    O(n,d)+= D(n,D) * E(d,D);
        //}
    }
}
#undef D
#undef O
#undef E
#undef M
return out;
}


void dsift(float *im, int rows, int cols, PHOWopts opts, int *numFrames, int *descrSize, float **framesOut, float **descrsOut)
{
    /* We use double pointer because they are lists of pointers and we want to modify them so they are visible outside*/
    
    /* Prepare vars */
    
    VlDsiftKeypoint const *frames ;
    VlDsiftDescriptorGeometry const *geom ;
    float const *descrs ;
    
    
    /* Init sift, set bounds, and flat window */
    VlDsiftFilter *dsift = vl_dsift_new_basic (rows, cols, opts.step, opts.currentSize);
    vl_dsift_set_bounds(dsift,
            VL_MAX(opts.bounds[1]-1, 0),
            VL_MAX(opts.bounds[0]-1, 0),
            VL_MIN(opts.bounds[3], rows - 1),
            VL_MIN(opts.bounds[2], cols - 1));
    int useFlatWindow = opts.fast;
    vl_dsift_set_flat_window(dsift, useFlatWindow) ;
    
    /* Window size */
    if (opts.windowSize >= 0) {
        vl_dsift_set_window_size(dsift, opts.windowSize) ;
    }
    *numFrames = vl_dsift_get_keypoint_num (dsift) ;
    *descrSize = vl_dsift_get_descriptor_size (dsift) ;
    geom = vl_dsift_get_geometry (dsift) ;
    vl_dsift_process (dsift, im) ;
    
    frames = vl_dsift_get_keypoints (dsift) ;
    descrs = vl_dsift_get_descriptors (dsift);
    
    
    *descrsOut = (float*) malloc(*numFrames* *descrSize*sizeof(float));
    float *tmpDescr = (float*)malloc(sizeof(float) * *descrSize) ;
    
    *framesOut =(float*) malloc(*numFrames * (opts.norm?4:3) * sizeof(float));
    
    float *pFrames = *framesOut;
    float *pDescrs = *descrsOut;
    
    for (int k=0; k < *numFrames; k++)
    {
        /* Copy frames */
        *pFrames++ = frames[k].y + 1;
        *pFrames++ = frames[k].x + 1;
        if (opts.norm)
            *pFrames++ = frames[k].norm;
        *pFrames++ = opts.currentSize;
        
        /* Copy descr */
        vl_dsift_transpose_descriptor (tmpDescr,
                descrs + *descrSize * k,
                geom->numBinT,
                geom->numBinX,
                geom->numBinY) ;
        if (opts.floatDescriptors)
        {
            for (int i=0; i < *descrSize;i++)
            {
                *pDescrs++ = VL_MIN(512.0f* tmpDescr[i],255.0F);
            }
        }
        else
        {
            for (int i=0; i < *descrSize;i++)
            {
                *pDescrs++ = floor(VL_MIN(512.0f* tmpDescr[i],255.0F));
            }
        }
    }
    free(tmpDescr);
    vl_dsift_delete (dsift) ;
}


void imsmooth_(float *outputImage, size_t numOutputRows, size_t numOutputColumns, float *inputImage, size_t numRows, size_t numColumns, size_t numChannels, int kernel, float sigma, int step, int flags)

{
    float * tempImage = (float*) malloc (sizeof(float) * numRows * numOutputColumns) ;
    int k ;
    int j ;
    
    /* Note that MATLAB uses a column major ordering, while VLFeat a row
     * major (standard) ordering for the image data. Effectively, VLFeat
     * is operating on a transposed image, but this is fine since filters
     * are symmetric.
     *
     * Therefore:
     *
     * input image width  = numRows
     * input image height = numColumns
     * output image width = numOutputRows (downsamped rows)
     * outout image height = numOutputColumns (downsampled columns)
     *
     * In addition a temporary buffer is used. This is an image that
     * is obtained from the input image by convolving and downsampling
     * along the height and saving the result transposed:
     *
     * temp image width  = numOutputColumns
     * temp image height = numRows
     */
    
    switch (kernel) {
        case GAUSSIAN :
        {
            size_t W = ceil (4.0 * sigma) ;
            float * filter = (float*) malloc (sizeof(float) * (2 * W + 1)) ;
            float acc = 0 ;
            for (j = 0 ; j < (signed)(2 * W + 1) ; ++j) {
                float z = ( (float) j - W) / (sigma + VL_EPSILON_F) ;
                filter[j] = exp(- 0.5 * (z*z)) ;
                acc += filter[j] ;
            }
            for (j = 0 ; j < (signed)(2 * W + 1) ; ++j) {
                filter[j] /= acc ;
            }
            
            for (k = 0 ; k < numChannels ; ++k) {
                
                vl_imconvcol_vf(tempImage, numOutputColumns,
                        inputImage, numRows, numColumns, numRows,
                        filter, -W, W, step, flags) ;
                
                vl_imconvcol_vf(outputImage, numOutputRows,
                        tempImage, numOutputColumns, numRows, numOutputColumns,
                        filter, -W, W, step, flags) ;
                
                inputImage += numRows * numColumns ;
                outputImage += numOutputRows * numOutputColumns ;
            }
            free (filter) ;
            break ;
        }
        
        case TRIANGULAR:
        {
            unsigned int W = VL_MAX((unsigned int) sigma, 1) ;
            for (k = 0 ; k < numChannels ; ++k) {
                
                vl_imconvcoltri_f(tempImage, numOutputColumns,
                        inputImage, numRows, numColumns, numRows,
                        W, step, flags) ;
                
                vl_imconvcoltri_f(outputImage, numOutputRows,
                        tempImage, numOutputColumns, numRows, numOutputColumns,
                        W, step, flags) ;
                
                inputImage += numRows * numColumns ;
                outputImage += numOutputRows * numOutputColumns ;
            }
            break ;
        }
        
    }
    free (tempImage) ;
}




float *imsmooth(img_t *image,float sigma, int *newRows, int *newCols)
{
    float *dst;
    int padding = VL_PAD_BY_CONTINUITY ;
    int kernel = GAUSSIAN ;
    int flags ;
    
    int cols = image->W;
    int rows = image->H;
    
    size_t step = 1 ;
    
    
    size_t M, N, K, M_, N_;
    
    M = rows;
    N = cols;
    K = 1;
    
    if ( (sigma < 0.01) && (step == 1))
    {
        dst = (float*)malloc(cols*rows*sizeof*dst);
        memcpy(dst, image->data, sizeof(float)*cols*rows);
        *newCols = cols;
        *newRows = rows;
        return dst;
    }
    
    M_ = (M-1)/step +1;
    N_ = (N-1)/step +1;
    
    
    
    dst  = (float*)malloc(M_*N_*sizeof*dst);
    flags  = padding ;
    flags |= VL_TRANSPOSE ;
    
    imsmooth_((float*) dst,
            M_, N_,
            image->data,
            M, N, K,
            kernel, sigma, step, flags) ;
    *newRows = M_;
    *newCols = N_;
    return dst;
}


void fillWithNans(FILE *f, int N)
{
    
    int maxBuffer = 10000000;
    int nBatches = (int)ceil((float)N/maxBuffer);
    float *nanbuf = (float*)malloc(maxBuffer * sizeof(float));
    float nan = (float)mxGetNaN();
    for (int i=0; i < maxBuffer; i++)
        nanbuf[i] = nan;
    
    int written = 0;
    for (int cb = 0; cb < nBatches;cb++)
    {
        int toWrite = (N - written)> maxBuffer?maxBuffer:(N - written);
        fwrite(nanbuf, sizeof(float), toWrite, f);
        written += toWrite;
    }
    free(nanbuf);
}


void PHOW(img_t *image, int step, int *sizes, int nSizes, int *nFrames, int *descrSize, float **framesOut, float **descrsOut)
{
    /* Default options */
    PHOWopts opts;
    
    opts.fast=1;
    opts.floatDescriptors=1;
    opts.sizes = sizes;
    opts.nsizes = nSizes;
    opts.step=step;
    opts.magnif=6;
    opts.windowSize=1.5;
    opts.currentSize=opts.sizes[0];
    opts.contrastthreshold = 0.005 ;
    /* Set norm to 1, otherwise it is not possible to clean the descriptors with low contrast */
    opts.norm = 1;
    
    /* Get max scale */
    int maxScale = max_int(opts.sizes, nSizes);
    
    /* Do the thing at each scale */
    
    int *numFrames = (int*)malloc(sizeof(int)*opts.nsizes);
    
    float **framesTmp = (float**)malloc(sizeof(float*) * opts.nsizes);
    float **descrsTmp = (float**)malloc(sizeof(float*) * opts.nsizes);
    
    for (int scale = 0; scale < opts.nsizes; scale++)
    {
        opts.currentSize = opts.sizes[scale];
        /* Smooth the image*/
        float sigma = opts.sizes[scale] / opts.magnif;
        int newCols, newRows;
        float *ims = imsmooth(image, sigma, &newRows, &newCols);
        
        
        /*
         * char output[64];
         * sprintf(output, "smoothed_%d.bin",scale);
         * writeMat(output, ims, newCols,newRows );
         */
        
        /* Start the show */
        
        /* Get offset. Warning, original is in matlab format, off by one... */
        int off = floor(1+ (3/2.0) * (maxScale - opts.sizes[scale]));
        
        //int off = floor( (3/2.0) * (maxScale - opts.sizes[scale]));
        opts.bounds[0]=off;
        opts.bounds[1]=off;
        opts.bounds[2]=INF;
        opts.bounds[3]=INF;
        
        /* Do sift */
        dsift(ims, newRows, newCols, opts, &numFrames[scale], descrSize, &framesTmp[scale], &descrsTmp[scale]);
        free(ims);
    }
    *nFrames=0;
    
    /* Keep only sifts that have a contrast > threshold. Build a list of indexes to keep for each level and counters */
    int **toKeep = (int**)malloc(sizeof(int*) * opts.nsizes);
    int *NtoKeep = (int*)malloc(sizeof(int) * opts.nsizes);
    for (int scale = 0; scale < opts.nsizes; scale++)
    {
        NtoKeep[scale]=0;
        int t =numFrames[scale];
        toKeep[scale] = (int*)malloc(sizeof(int)*t);
        for (int p=0; p < t; p++)
        {
            if (framesTmp[scale][4*p+2] >=opts.contrastthreshold)
            {
                toKeep[scale][NtoKeep[scale]]=p;
                NtoKeep[scale]++;
            }
        }
        *nFrames+=NtoKeep[scale];
    }
    
    /* And now copy only the relevant ones */
    *framesOut = (float *) malloc(4* *nFrames*sizeof(float));
    *descrsOut = (float *) malloc(*descrSize* *nFrames*sizeof(float));
    float *pFrames = *framesOut;
    float *pDescrs = *descrsOut;
    for (int scale = 0; scale < opts.nsizes; scale++)
    {
        for (int p=0; p < NtoKeep[scale];p++)
        {
            memcpy(pFrames, &framesTmp[scale][4*toKeep[scale][p]], 4*sizeof(float));
            memcpy(pDescrs, &descrsTmp[scale][*descrSize*toKeep[scale][p]], *descrSize*sizeof(float));
            pFrames+=4;
            pDescrs+=*descrSize;
        }
        
    }
    
    
    
    /* Divide by 255, square root, and normalize with a 0.25 norm */
    float s=0;
    pDescrs=*descrsOut;
    float *pDescrs2=*descrsOut;
    for (int i=0; i < *nFrames;i++)
    {
        /* Do division and square root and accumulate norm */
        pDescrs2 = pDescrs;
        s=0;
        for (int j=0; j < *descrSize;j++){
            *pDescrs = sqrt(*pDescrs/255);
            s+=(*pDescrs * *pDescrs);
            pDescrs++;
        }
        /* Get the 0.25 and invert*/
        s = 1 / sqrt(sqrt(s));
        /* multiply by the inverse norm*/
        for (int j=0; j < *descrSize;j++){
            *pDescrs2++*=s;
        }
    }
    
    
    /* Free stuff */
    for (int i=0; i < opts.nsizes; i++)
    {
        free(framesTmp[i]);
        free(descrsTmp[i]);
        free(toKeep[i]);
    }
    free(numFrames);
    free(framesTmp);
    free(descrsTmp);
    free(toKeep);
    free(NtoKeep);
    
}


void getFV(img_t *image,pca_t *pca, gmm_t *gmm, int step, int * sizes, int nSizes, int doMiniBox, float *fv)
{
    int fvD = 2* gmm->D * gmm->G;
    /* Extract phow */
    float *frames;
    float *descrs;
    int nFrames;
    int descrSize;
    
    /* PHOW */
    PHOW(image, step, sizes, nSizes, &nFrames, &descrSize, &frames, &descrs);
    
    /* PCA */
    float *projected = ApplyPca(descrs,nFrames, descrSize, pca->d, pca);
    free(descrs);
    /* Append coordinates (minibox?) */
    boundingBox_t box;
    if (!doMiniBox)
    {
        /* box is the whole image */
        box.x=0; box.y=0;
        box.w = image->W;
        box.h = image->H;
        box.cx = (int)round(box.w/2.0);
        box.cy = (int)round(box.h/2.0);
    }
    else
    {
        extractWordBox(image,0.975,0.8, &box);
        
    }
    
    float *descrxy = appendXY(projected,  frames, nFrames, pca->d,  &box);
    
    free(projected);
    free(frames);
    
    
    /* Compute fv */
    vl_fisher_encode(fv, VL_TYPE_FLOAT,gmm->mu, gmm->D, gmm->G, gmm->sigma, gmm->w, descrxy, nFrames,VL_FISHER_FLAG_IMPROVED);
}

long int * readImagesToc(char *imagesPath, int *nImages)
{
    FILE *f = fopen(imagesPath, "rb");
    fread(nImages, sizeof(int), 1, f);
    long int *toc = (long int*)malloc(*nImages*sizeof(long int));
    fread(toc, sizeof(long int), *nImages,f);
    fclose(f);
    return toc;
}

img_t *readAndConvertImage(FILE *fp)
{
    
    img_t *im = (img_t*)malloc(sizeof(img_t));
        fread(&(im->W), sizeof(int), 1, fp);
        fread(&(im->H), sizeof(int), 1, fp);
        int nd = im->W * im->H;
        im->data = (float*)malloc(nd *sizeof(float));
        unsigned char *buffer = (unsigned char *)malloc(nd *sizeof(unsigned char));
        fread(buffer, sizeof(unsigned char), nd, fp);
        for (int i=0; i < nd;i++)
            im->data[i] = ((float)buffer[i])/255.0;
        free(buffer);
        return im;
}

img_t** readImagesBatch(char *imagesPath, long int *toc, int start, int end)
{
    int nImages = end-start;
    img_t** images = (img_t**)malloc(nImages*sizeof(*images));
    FILE *f = fopen(imagesPath, "rb");
    for (int i=start; i < end; i++)
    {
        fseek(f, toc[i], SEEK_SET);
        images[i-start] = readAndConvertImage(f);
    }    
    fclose(f);
    return images;
}




void processImages(char *imagesPath,pca_t * pca, gmm_t *gmm, int step, int *sizes, int nSizes, int doMinibox, char * outPath)
{
        
    int nImages;
    long int *toc;
    toc = readImagesToc(imagesPath,  &nImages);
    img_t **images;
    
    int fvdim = 2*gmm->D * gmm->G;
    /* Open the output, fill with nans, and close */
    FILE *f = fopen(outPath, "wb");
    fwrite(&nImages, sizeof(int), 1, f);
    fwrite(&fvdim, sizeof(int), 1, f);
    fillWithNans(f, fvdim*nImages);
    fclose(f);
    
    int imagesPerBatch = 1000;
    int nBatches = (int)ceil(((float)nImages)/imagesPerBatch);
    
    float *buffer = (float*)malloc(imagesPerBatch * fvdim * sizeof(float));
    
    
    for (int cb=0; cb < nBatches; cb++)
    {
        mexPrintf("Batch %d/%d\n",cb,nBatches);
        int sp = cb * imagesPerBatch;
        int ep = (cb+1)*imagesPerBatch > nImages?nImages:(cb+1)*imagesPerBatch;
        int nInBatch = ep-sp;
        images = readImagesBatch(imagesPath, toc, sp,ep);
        
        
        #pragma omp parallel for
        for (int cf = 0; cf < nInBatch; cf++)
        {
            getFV(images[cf+sp], pca, gmm, step, sizes, nSizes, doMinibox, &buffer[cf*fvdim]);
            
        }
        
        /* Dump the current fvs */
        f = fopen(outPath, "r+");
        fseek(f, 2*sizeof(int)  + cb*imagesPerBatch*fvdim * sizeof(float), SEEK_SET);
        fwrite(buffer, sizeof(float), nInBatch*fvdim, f);
        fclose(f);
        
        freeImages(&images,nInBatch);
    }
    free(toc);
    free(buffer);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *imagesPath = mxArrayToString(prhs[0]);
    char *PCAPath = mxArrayToString(prhs[1]);
    char *GMMPath = mxArrayToString(prhs[2]);
    int step = (int)mxGetScalar(prhs[3]);
    int *sizes = (int*)mxGetData(prhs[4]);
    int nSizes = mxGetN(prhs[4]);
    int doMinibox = (int)mxGetScalar(prhs[5]);
    char *outPath = mxArrayToString(prhs[6]);

    pca_t *pca = readPCA(PCAPath);
    gmm_t *gmm = readGMM(GMMPath);
    
    
    processImages(imagesPath, pca, gmm, step, sizes, nSizes, doMinibox, outPath);
    
    
    
    freePCA(&pca);
    freeGMM(&gmm);
    return;
}

