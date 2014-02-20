function [mAP, rec] = compute_mAP(resultLabels, numRelevant)

resultLabels = single(resultLabels);
resultLabels(resultLabels<0) = 0;
[r,c] = size(resultLabels);


if r>1 && c>1
    precAt = cumsum(resultLabels)./repmat([1:size(resultLabels,1)]',1,size(resultLabels,2));
    mAP = sum(precAt.*resultLabels)/numRelevant;
    rec= sum(resultLabels)/numRelevant;
    return;
end 

if r>c
    resultLabels = resultLabels';
end


% Compute the mAP
numRelevant = single(numRelevant);
precAt = cumsum(resultLabels)./[1:length(resultLabels)];
mAP = sum(precAt.*resultLabels)/numRelevant;
rec= sum(resultLabels)/numRelevant;
end