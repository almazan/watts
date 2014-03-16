function [ count ] = writeMat(mat,f)
[D,N] = size(mat);
fid = fopen(f, 'w');
fwrite(fid, N, 'int32');
fwrite(fid, D, 'int32');
count=fwrite(fid, mat, 'single');
fclose(fid);
end

