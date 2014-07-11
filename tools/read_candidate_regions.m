function [candidates,docsDic] = read_candidate_regions(path,wordsGT,classes)
% if nargin==0
%     path = '~/Development/SVT_cropped_bboxes/';
% end
% load('SVT_wordsGT.mat');

d = dir(path);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

c = 0;
docsDic = containers.Map();
idxDoc = 0;
for i=1:length(nameFolds)
    
    fid = fopen(fullfile(path,nameFolds{i},'info.txt'),'r');
    if fid>0
        text = textscan(fid,'%s %d %d %d %d %f %s');
        imId = classes([nameFolds{i} '.jpg']);
        for j=1:length(text{1,1})
            file = fullfile(path,nameFolds{i},text{1}{j});
            c = c+1;
            candidates(c).im = rgb2gray(imread(file));
            candidates(c).fname = text{1}{j};
            candidates(c).imId = imId;
            candidates(c).x1 = text{2}(j);
            candidates(c).y1 = text{3}(j);
            candidates(c).w = text{4}(j);
            candidates(c).h = text{5}(j);
            candidates(c).overlap = text{6}(j);
            candidates(c).x2 = candidates(c).x1+candidates(c).w-1;
            candidates(c).y2 = candidates(c).y1+candidates(c).h-1;
            candidates(c).imname = nameFolds{i};
            if strcmp(text{7}{j},'') || candidates(c).overlap<0.5
                candidates(c).gttext = '-';
            else
                candidates(c).gttext = text{7}{j};
            end
            if ~docsDic.isKey(nameFolds{i})
                idxDoc = idxDoc+1;
                docsDic(nameFolds{i}) = idxDoc;
                candidates(c).docId = idxDoc;
            else
                candidates(c).docId = docsDic(nameFolds{i});
            end
        end
    end
    fclose(fid);
end

relBoxes = [wordsGT(:).x1; wordsGT(:).y1; wordsGT(:).w; wordsGT(:).h]';
imIds = [wordsGT(:).imId];
dif1 = 0;
dif2 = 0;
for i=1:length(candidates)
    predBox = [candidates(i).x1; candidates(i).y1; candidates(i).w; candidates(i).h]';
    
    intArea = rectint(single(predBox), single(relBoxes));
    
    areaP = single(candidates(i).h*candidates(i).w);
    areaGt = single(([wordsGT(:).h].*[wordsGT(:).w])');
    
    denom = bsxfun(@minus, single(areaP+areaGt'), intArea);
    overlap = intArea./denom;
    
    [y,x] = find(overlap >= 0.5 & imIds==candidates(i).imId);
    if ~isempty(x)
        if strcmp(candidates(i).gttext,'-')
            dif1 = dif1+1;
        end
        candidates(i).wId = x;
        candidates(i).gttext = wordsGT(x).gttext;
    else
        if ~strcmp(candidates(i).gttext,'-')
            dif2 = dif2+1;
        end
        candidates(i).wId = -1;
        candidates(i).gttext = '-';
    end
end

end