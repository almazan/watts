function [p1, mAP, qidx] = eval_dp_asymm(opts, queries, dataset, allClasses, queriesWords, doqbs)
if nargin < 6
    doqbs = 0;
end

% Get size and prepare stuff
[d,N] = size(queries);
queries = single(queries');
dataset = single(dataset');

%For each element, find the number of relevants
H = hist(single(allClasses), single(1:max(max(allClasses))));
nRel = H(allClasses)';

% Keep as queries only words that appear at least 1 more time
% besides being the query. If necessary, remove stopwords)
keep = find(nRel >=0);
if opts.RemoveStopWords
    sw = textread(opts.swFile,'%s','delimiter',',');
    sw = unique(lower(sw));
    keep = find(~ismember(lower(queriesWords), lower(sw)));
end
if doqbs == 0;
    keep = intersect(keep, find(nRel >=2));
else
    keep = intersect(keep, find(nRel >=1));
end

% If QBS, keep only one instance per query
if doqbs
    set = containers.Map('keytype','int32','ValueType','int32');
    % If qbe, keep only one instance of each class as query
    keep2=[];
    for i=1:length(keep)
        if ~set.isKey(allClasses(keep(i)))
            set(allClasses(keep(i)))=1;
            keep2 = [keep2 keep(i)];
        end
    end
    keep = keep2;
end
keepQueries = keep;

% Get updated queries
queries = queries(keepQueries,:);
queriesClases = allClasses(keepQueries);

% Keep as queries only words also occurring in training
% load(opts.Data,'wordsTr');
% keep = ismember(lower(queriesWords), lower(unique(wordsTr)));
% queries = queries(keep,:);
% queriesClases = queriesClases(keep);

N = size(queries,1);

% Compute the number of relevants for each query:
NRelevantsPerQuery = H(queriesClases)';

if doqbs
    s = sum(queries,2);
    idxNan = find(isnan(s));
end

queries(isnan(queries)) = 0;

% Compute scores
S=queries*dataset';

% The last part automatically removes the query from the dataset when
% computing stats by providing the indexes of the queries. If the queries
% are not included in the dataset, set the last parameter to a vector of
% [-1,-1,...,-1];
if doqbs==0
[p1, mAP,bestIdx] = computeStats_c(single(S'),int32(queriesClases), int32(allClasses), int32(NRelevantsPerQuery), int32(keepQueries'-1));
else
    v = ones(N)*-1;
[p1, mAP,bestIdx] = computeStats_c(single(S'),int32(queriesClases), int32(allClasses), int32(NRelevantsPerQuery), int32(v));
end

qidx=keepQueries;


end
