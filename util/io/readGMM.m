function [ GMM ] = readGMM(f)
[fid,msg] = fopen(f, 'r');
G=fread(fid, 1, '*int32');
D=fread(fid, 1,'*int32');
GMM.we = fread(fid, [1,G], '*single');
GMM.mu = fread(fid, [D,G], '*single');
GMM.sigma = fread(fid, [D,G], '*single');
fclose(fid);
end
