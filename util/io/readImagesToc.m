function toc = readImagesToc(f)
[fid,msg] = fopen(f, 'r');
nImages=fread(fid, 1, '*int32');
toc = fread(fid, nImages, '*int64');
fclose(fid);
end
