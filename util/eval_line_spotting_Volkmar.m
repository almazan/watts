function map = eval_line_spotting_Volkmar(opts, queries, dataset, queriesClasses, datasetClasses, queriesWords, linesTe, line2classes)
% linesTe = DATA.linesTe;
% allClasses = DATA.queriesClassesTe;
% queriesWords = DATA.wordsTe;

% doqbs = 1;
queries = single(queries');
dataset = single(dataset');

%For each element, find the number of relevants
% H = hist(single(allClasses), single(1:max(max(allClasses))));
% nRel = H(allClasses)';

% Keep as queries only words that appear at least 1 more time
% besides being the query. If necessary, remove stopwords)
% keep = find(nRel >=0);
if opts.RemoveStopWords
    sw = textread(opts.swFile,'%s','delimiter',',');
    sw = unique(lower(sw));
    keep = find(~ismember(lower(queriesWords), lower(sw)));
end



% set = containers.Map('keytype','int32','ValueType','int32');
% % If qbe, keep only one instance of each class as query
% keep2=[];
% for i=1:length(keep)
%     if ~set.isKey(queriesClasses(keep(i)))
%         set(allClasses(keep(i)))=1;
%         keep2 = [keep2 keep(i)];
%     end
% end
% keep = keep2;

keepQueries = keep;

queries = queries(keepQueries,:);
N = size(queries,1);

% Compute scores
S=queries*dataset';

queriesClases = queriesClasses(keepQueries);

% Compute the number of relevants for each query:
% NRelevantsPerQuery = H(queriesClases)';

N = size(queries,1);
% N2 = size(allClasses);
map = zeros(N,1);
[V,I] = sort(S,2,'descend');
% results = zeros(N2,1);
remove = [];
for i=1:N
%     nRel = NRelevantsPerQuery(i);
    class = queriesClases(i);
    nRel = sum(datasetClasses==class);
    if nRel==0
        remove = [remove i];
        continue;
    end
    ind = I(i,:);
%     results = datasetClasses==class;
    lineUsed = zeros(length(linesTe),1);
    n=0;
    results = [];
    for j=1:length(ind)
        if lineUsed(ind(j))==0 && n<nRel
            lineUsed(ind(j)) = 1;
            line = linesTe{ind(j)};
            classes = line2classes(line);
            res = sum(classes==class)>0;
            results = [results res];
            if res == 1
                n = n+1;
            end
        end
    end
%     nRel = sum(results);
    map(i) = compute_mAP(results,nRel); 
end
map(remove) = [];
end