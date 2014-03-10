function [ image ] = readImage( fid,toc,i )
%SAVEIMAGES Summary of this function goes here
%   Detailed explanation goes here

fseek(fid, toc(i), 'bof');
N=fread(fid, 1, 'int32');
D=fread(fid, 1,'int32');
image = fread(fid, [D,N], '*uint8');
end



