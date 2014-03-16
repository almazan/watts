function [ count ] = writePCA(PCA,f)
[D,N] = size(PCA.eigvec);
fid = fopen(f, 'w');
fwrite(fid, N, 'int32');
fwrite(fid, D, 'int32');
count=fwrite(fid, PCA.eigvec, 'single');
count=fwrite(fid, PCA.mean, 'single');
fclose(fid);
end

