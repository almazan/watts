function [ count ] = writeCCA(CCA,f)
Wx = CCA.Wx(:,1:CCA.K);
Wy = CCA.Wy(:,1:CCA.K);
[N,K] = size(Wx);
fid = fopen(f, 'w');
fwrite(fid, N, 'int32');
fwrite(fid, K, 'int32');
count=fwrite(fid, Wx, 'single');
count=fwrite(fid, Wy, 'single');
count=fwrite(fid, CCA.matts, 'single');
count=fwrite(fid, CCA.mphocs, 'single');
fclose(fid);
end

