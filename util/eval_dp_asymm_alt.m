function [p1, mAP, qidx] = eval_dp_asymm_alt(queries, dataset, classesTe, classesTr, doqbs)
%doqbe=0;
if nargin < 5
    doqbs = 0;
end

% Get size and prepare stuff
[d,N] = size(queries);
queries = single(queries');
dataset = single(dataset');

%For each element, find the number of relevants
% H = hist(single(classesTr), single(1:max(max(classesTr))));
% nRel = H(classesTr)';

NRelevantsPerQuery = zeros(length(classesTe),1);
for i=1:length(classesTe)
    NRelevantsPerQuery(i) = sum(classesTe(i)==classesTr);
end

keep = find(NRelevantsPerQuery >=1);
% If QBS, keep only one instance per query
if doqbs
    set = containers.Map('keytype','int32','ValueType','int32');
    % If qbe, keep only one instance of each class as query
    keep2=[];
    for i=1:length(keep)
        if ~set.isKey(classesTe(keep(i)))
            set(classesTe(keep(i)))=1;
            keep2 = [keep2 keep(i)];
        end
    end
    keep = keep2;
end
keepQueries = keep;

% Get updated queries
queries = queries(keepQueries,:);
N = size(queries,1);

% Compute scores
S=queries*dataset';

queriesClases = classesTe(keepQueries);
NRelevantsPerQuery = NRelevantsPerQuery(keepQueries);

% Compute the number of relevants for each query:
% NRelevantsPerQuery = H(queriesClases)';

% The last part automatically removes the query from the dataset when
% computing stats by providing the indexes of the queries. If the queries
% are not included in the dataset, set the last parameter to a vector of
% [-1,-1,...,-1];

v = ones(size(keepQueries))*-1;
[p1, mAP] = computeStats_c(single(S'),int32(queriesClases), int32(classesTr), int32(NRelevantsPerQuery), int32(v));

qidx=keepQueries;


end
