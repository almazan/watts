function [ count ] = writeImages( imagesCell, f )
%SAVEIMAGES Summary of this function goes here
%   Detailed explanation goes here

nImages = length(imagesCell);
fid = fopen(f, 'w');
fwrite(fid, nImages, 'int32');
for i=1:nImages
    im = imagesCell{i};
    [D,N] = size(im);
    fwrite(fid, N, 'int32');
    fwrite(fid, D, 'int32');
    count=fwrite(fid, im2single(im), 'single');
end
end

