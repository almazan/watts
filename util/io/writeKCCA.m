function [ count ] = writeKCCA(KCCA,f)
K = KCCA.K;
N = size(KCCA.rndmatx,2);
M = KCCA.M;
fid = fopen(f, 'w');
fwrite(fid, N, 'int32');
fwrite(fid, K, 'int32');
fwrite(fid, M, 'int32');
count=fwrite(fid, KCCA.rndmatx, 'single');
count=fwrite(fid, KCCA.rndmaty, 'single');
count=fwrite(fid, KCCA.Wx, 'single');
count=fwrite(fid, KCCA.Wy, 'single');
count=fwrite(fid, KCCA.matts, 'single');
count=fwrite(fid, KCCA.mphocs, 'single');
fclose(fid);
end

