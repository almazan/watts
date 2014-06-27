function [ CCA ] = readCCA(f)
[fid,msg] = fopen(f, 'r');
N=fread(fid, 1, '*int32');
K=fread(fid, 1,'*int32');
CCA.Wx = fread(fid, [N,K], '*single');
CCA.Wy = fread(fid, [N,K], '*single');
CCA.matts = fread(fid, [N,1], '*single');
CCA.mphocs = fread(fid, [N,1], '*single');
CCA.K = K;
fclose(fid);
end