folderImages = '~/Downloads/sample_images/';
folderData = 'data/SVT_noBigrams';

d = dir(fullfile(folderImages,'*.jpg'));

for i=1:length(d)
    name = d(i).name;
    images{i} = imread(fullfile(folderImages,name));
end

atts = image2att(images,folderData);

for i=1:length(d)
    name = d(i).name;
    name = [name(1:end-4) '.bin'];
    writeMat(atts(:,i),fullfile(folderImages,name));
end