function [ count ] = writeGMM(GMM,f)
[D,G] = size(GMM.mu);
fid = fopen(f, 'w');
fwrite(fid, G, 'int32');
fwrite(fid, D, 'int32');
count=fwrite(fid, GMM.we, 'single');
count=fwrite(fid, GMM.mu, 'single');
count=fwrite(fid, GMM.sigma, 'single');
fclose(fid);
end

