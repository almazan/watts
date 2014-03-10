function [ PCA ] = readPCA(f)
[fid,msg] = fopen(f, 'r');
N=fread(fid, 1, '*int32');
D=fread(fid, 1,'*int32');
PCA.eigvec = fread(fid, [D,N], '*single');
PCA.mean = fread(fid, [D,1], '*single');
fclose(fid);
end
