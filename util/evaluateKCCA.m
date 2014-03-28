function  mAP = evaluateKCCA(opts,DATA,embedding)
% Evaluate KCCA.
% Small fix for versions of matlab older than 2012b ('8') that do not support stable intersection
if verLessThan('matlab', '8')
    inters=@stableintersection;
else
    inters=@intersect;
end

fprintf('\n');
disp('**************************************');
disp('***************  KCSR   **************');
disp('**************************************');

matx = embedding.rndmatx(1:embedding.M,:);
maty = embedding.rndmaty(1:embedding.M,:);

tmp = matx*DATA.attReprTe;
attReprTe_emb = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
tmp = maty*DATA.phocsTe;
phocsTe_emb = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];

% Mean center
attReprTe_emb=bsxfun(@minus, attReprTe_emb, embedding.matts);
phocsTe_emb=bsxfun(@minus, phocsTe_emb, embedding.mphocs);

% Embed test
attReprTe_cca = embedding.Wx(:,1:embedding.K)' * attReprTe_emb;
phocsTe_cca = embedding.Wy(:,1:embedding.K)' * phocsTe_emb;

% L2 normalize (critical)
attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
qbs_test_map = mean(mAPEucl);
qbs_test_p1 = mean(p1);


% Display info
disp('------------------------------------');
fprintf('reg: %.8f. k: %d\n',  embedding.reg, embedding.K);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

mAP.qbe = 100*qbe_test_map;
mAP.qbs = 100*qbs_test_map;


%% Eval test vs train QBE
if strcmpi(opts.dataset,'IIIT5K')
    attReprTr = readMat(opts.fileAttRepresTr);
    tmp = matx*attReprTr;
    attReprTr_cca = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
    attReprTr_cca=bsxfun(@minus, attReprTr_cca, embedding.matts);
    attReprTr_cca = embedding.Wx(:,1:embedding.K)' * attReprTr_cca;
    attReprTr_cca = (bsxfun(@rdivide, attReprTr_cca, sqrt(sum(attReprTr_cca.*attReprTr_cca))));
    
    [p1,mAPEucl, q] = eval_dp_asymm_alt(attReprTe_cca,attReprTr_cca,DATA.wordClsTe,DATA.wordClsTr);
    qbe_test_map = mean(mAPEucl);
    qbe_test_p1 = mean(p1);
    
    fprintf('\n');
    disp('------------------------------------');
    fprintf('Test vs Train');
    fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
    disp('------------------------------------');
    
    mAP.alt_test_vs_train.qbe = 100*qbe_test_map;
end

%% Eval with words only appearing in training
% QBS (note the 1 at the end)
if strcmpi(opts.dataset,'GW')
    idx = ismember(DATA.wordClsTe,DATA.wordClsTr);
    phocsTe_cca = phocsTe_cca(:,idx);
    queriesCls = DATA.wordClsTe(idx);
    [p1,mAPEucl, q] = eval_dp_asymm_alt(phocsTe_cca,attReprTe_cca,queriesCls,DATA.wordClsTe,1);
    qbs_test_map = mean(mAPEucl);
    qbs_test_p1 = mean(p1);
    
    % Display info
    disp('------------------------------------');
    fprintf('Evaluation with queries that appear in training\n');
    fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
    disp('------------------------------------');
    
    mAP.alt_only_training_words.qbs = 100*qbs_test_map;
end

%% Eval line spotting
if strcmpi(opts.dataset,'IAM')
    load('data/IAM_dict_line2classes.mat');
    DATA.line2classes = line2classes;
    
    % Preparing queries for protocol
    words = [DATA.labelsTr DATA.labelsVa DATA.labelsTe];
    phocs = [DATA.phocsTr DATA.phocsVa DATA.phocsTe];
    wordCls = [DATA.wordClsTr DATA.wordClsVa DATA.wordClsTe];
    [u,ind,n] = unique(lower(words(:)));
    B = accumarray(n, 1, [], @sum);
    [B,I] = sort(B,'descend');
    words = u(I);
    ind = ind(I);
    [words,ia,ib] = inters(lower(words),lower([DATA.labelsTr DATA.labelsVa]),'stable');
    words = words(1:4000);
    ind = ind(ia);
    ind = ind(1:4000);
    phocs = phocs(:,ind);
    wordCls = wordCls(ind);
    tmp = maty*phocs;
    phocs_cca = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
    phocs_cca=bsxfun(@minus, phocs_cca, embedding.mphocs);
    phocs_cca = embedding.Wy(:,1:embedding.K)' * phocs_cca;
    phocs_cca = (bsxfun(@rdivide, phocs_cca, sqrt(sum(phocs_cca.*phocs_cca))));
    
    linesV = textread('data/test-lines.IAMTest_Vol.uniq.txt','%s','delimiter',',');
    idx = find(ismember(DATA.linesTe,linesV));
    
    mapLineSpotting = eval_line_spotting_Volkmar(opts, phocs_cca, attReprTe_cca(:,idx),wordCls,DATA.wordClsTe(idx),words,DATA.linesTe(idx),DATA.line2classes);
    
    mean(mapLineSpotting)
end

end

% Ugly hack to deal with the lack of stable intersection in old versions of
% matlab
function [r, ia, ib] = stableintersection(a, b, varargin)
[r,ia,ib] = intersect(a,b);
[ia, tmp2] = sort(ia);
ib = ib(tmp2);
r = r(tmp2);
end