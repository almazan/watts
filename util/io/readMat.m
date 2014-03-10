function [ mat ] = readMat(f)
[fid,msg] = fopen(f, 'r');
N=fread(fid, 1, '*int32');
D=fread(fid, 1,'*int32');
mat = fread(fid, [D,N], '*single');
fclose(fid);
end
