function [ KCCA ] = readKCCA(f)
[fid,msg] = fopen(f, 'r');
N=fread(fid, 1, '*int32');
K=fread(fid, 1,'*int32');
M=fread(fid, 1,'*int32');
KCCA.rndmatx = fread(fid, [M,N], '*single');
KCCA.rndmaty = fread(fid, [M,N], '*single');
KCCA.Wx = fread(fid, [M*2,K], '*single');
KCCA.Wy = fread(fid, [M*2,K], '*single');
KCCA.matts = fread(fid, [M*2,1], '*single');
KCCA.mphocs = fread(fid, [M*2,1], '*single');
KCCA.K = K;
KCCA.M = M;
fclose(fid);
end

