function [ images ] = readImages( f )
%SAVEIMAGES Summary of this function goes here
%   Detailed explanation goes here

[fid,msg] = fopen(f, 'r');
nImages=fread(fid, 1, '*int32');
toc = fread(fid, nImages, '*int64');
images = cell(1, nImages);
for i=1:nImages
    if toc(i)~=ftell(fid)
        disp(i);
    end
    N=fread(fid, 1, '*int32');
    D=fread(fid, 1,'*int32');
    images{i} = fread(fid, [D,N], '*uint8');
end
fclose(fid);
end



