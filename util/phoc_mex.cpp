#include "mex.h"
#include "string.h"
#include <string>

#include <map>

#define HARD 0


void computePhoc(char *str, std::map<char,int> vocUni2pos, std::map<std::string,int> vocBi2pos, int Nvoc, int *levels, int Nlevels, int totalLevels, float *out);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *pwords = prhs[0];
    const mxArray *pvoc = prhs[1];
    const mxArray *plevels = prhs[2];
    
    
    /* Read and copy words */
    int Nwords = mxGetNumberOfElements(pwords);
    char **words = (char**)mxMalloc(Nwords*sizeof(mxChar*));
    for (int i=0; i < Nwords;i++)
    {
        const mxArray *cellPtr = mxGetCell(pwords, i);
        int n = mxGetN(cellPtr);
        int buflen = n*sizeof(mxChar)+1;
        words[i] = (char*)mxMalloc(buflen);
        mxGetString(cellPtr,words[i],buflen);
    }
    
    /* Read and copy voc. Prepare dict */
    std::map<char,int> vocUni2pos;
    std::map<std::string,int> vocBi2pos;
    int Nvoc = mxGetNumberOfElements(pvoc);
    
    char **voc = (char**)mxMalloc(Nvoc*sizeof(mxChar*));
    for (int i=0; i < Nvoc;i++)
    {
        const mxArray *cellPtr = mxGetCell(pvoc, i);
        int n = mxGetN(cellPtr);
        int buflen = n*sizeof(mxChar)+1;
        voc[i] = (char*)mxMalloc(buflen);
        mxGetString(cellPtr,voc[i],buflen);
        
        if (n==1) vocUni2pos[*voc[i]]=i;
        if (n==2) {
            std::string str(voc[i]);
            vocBi2pos[str]=i;
        }
        if (n>=3)
        {
            mexPrintf("n-grams with n>=3 not supported. Ignoring %s\n",voc[i]);
        }
    }
    
    /* Read and copy levels */
    int *levels = (int*)mxGetData(plevels);
    int Nlevels = mxGetN(plevels);
    int totalLevels = 0;
    for (int i=0; i < Nlevels; i++)
    {
        totalLevels+=levels[i];
    }
    
    
    int phocSize = totalLevels*Nvoc;
    
    
    /* Prepare output */
    plhs[0] = mxCreateNumericMatrix(phocSize, Nwords,mxSINGLE_CLASS,mxREAL);
    float *phocs = (float*)mxGetData(plhs[0]);
    /* Compute */
    for (int i=0; i < Nwords;i++)
    {
        computePhoc(words[i], vocUni2pos, vocBi2pos,Nvoc, levels, Nlevels, totalLevels, &phocs[i*phocSize]);
    }
    
    
    /* Cleanup */
    for (int i=0; i < Nwords; i++)
        mxFree(words[i]);
    mxFree(words);
    for (int i=0; i < Nvoc; i++)
        mxFree(voc[i]);
    mxFree(voc);
}

void computePhoc(char *str, std::map<char,int> vocUni2pos, std::map<std::string,int> vocBi2pos, int Nvoc, int *levels, int Nlevels, int totalLevels, float *out)
{
    int phocSize = totalLevels*Nvoc;
    int strl = strlen(str);
    
    int doUnigrams = vocUni2pos.size()!=0;
    int doBigrams = vocBi2pos.size()!=0;
    
    /* For each block */
    float *p = out;
    for (int nl = 0; nl < Nlevels; nl++)
    {
        /* For each split in that level */
        for (int ns=0; ns < levels[nl]; ns++)
        {
            float starts = ns/(float)levels[nl];
            float ends = (ns+1)/(float)levels[nl];
            
            /* For each character */
            if (doUnigrams)
            {
                for (int c=0; c < strl; c++)
                {
                    if (vocUni2pos.count(str[c])==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        continue;
                    }
                    int posOff = vocUni2pos[str[c]];
                    float startc = c/(float)strl;
                    float endc = (c+1)/(float)strl;
                    
                    /* Compute overlap over character size (1/strl)*/
                    if (endc < starts || ends < startc) continue;
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl;
                    #if HARD
                    if (ov >=0.48)
                    {
                        p[posOff]+=1;
                    }
                    #else
                    p[posOff] = std::max(ov, p[posOff]);
                    #endif
                }
            }
            if (doBigrams)
            {
                for (int c=0; c < strl-1; c++)
                {
                    char back = str[c+2];
                    str[c+2]='\0';
                    std::string sstr(&str[c]);
                    if (vocBi2pos.count(sstr)==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        str[c+2]=back;
                        continue;
                    }
                    int posOff = vocBi2pos[sstr];
                    float startc = c/(float)strl;
                    float endc = (c+2)/(float)strl;
                    
                    /* Compute overlap over bigram size (2/strl)*/
                    if (endc < starts || ends < startc){ str[c+2]=back; continue;}
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl/2.0;
                    if (ov >=0.48)
                    {
                        p[posOff]+=1;
                    }
                    str[c+2]=back;
                }
            }
            p+=Nvoc;
        }
    }
    return;
}
