function [cer, p1, qidx] = compute_cer(queries, dataset, queriesClasses, queriesWords)

[allWords,uidx,unused] = unique(queriesWords);
allClasses = queriesClasses(uidx);
dataset = dataset(:,uidx);
% Remove those marked as 'trash'
keepClasses = find(~ismember(allWords, '-'));
allClasses = allClasses(keepClasses);
allWords = allWords(keepClasses);
dataset = dataset(:,keepClasses);


% Get size and prepare stuff
[d,N] = size(queries);
queries = single(queries');
dataset = single(dataset');

%For each element, find the number of relevants
H = hist(single(allClasses), single(1:max(max(allClasses))));
nRel = H(queriesClasses)';

% Remove as queries those marked as 'trash'
keepQueries = find(~ismember(queriesWords, '-'));

% Get updated queries
queries = queries(keepQueries,:);
N = size(queries,1);

% Compute scores
S=queries*dataset';

queriesClases = queriesClasses(keepQueries);

% Compute the number of relevants for each query:
NRelevantsPerQuery = H(queriesClases)';

% The last part automatically removes the query from the dataset when
% computing stats by providing the indexes of the queries. If the queries
% are not included in the dataset, set the last parameter to a vector of
% [-1,-1,...,-1];
idxs = int32(-ones(size(keepQueries, 1),1));
[p1, mAP, bestIdx] = computeStats_c(single(S'),int32(queriesClases), int32(allClasses), int32(NRelevantsPerQuery), idxs);    
qidx=keepQueries;

L=single(zeros(1,length(qidx)));
% Levenshtein only needed for the words that failed
badIdx = find(p1==0);
for i=badIdx'    
    L(i) = levenshtein_c(queriesWords{qidx(i)},allWords{bestIdx(i)});
end
cer = 100*mean(L);
               

end
