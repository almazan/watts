function [embedding_final,embRepr_final] = learn_common_subspace(opts,data)


%% Load attribute representations
load(opts.fileAttRepres);
data.attReprTr = single(attReprTr);
data.attReprTe = single(attReprTe);
data.phocsTr = single(data.phocsTr);
data.phocsTe = single(data.phocsTe);
data.wordClsTr = single(data.wordClsTr);
data.wordClsTe = single(data.wordClsTe);

keep = sum(data.phocsTr)~=0;
data.attReprTr = data.attReprTr(:,keep);
data.phocsTr = data.phocsTr(:,keep);
data.wordClsTr = data.wordClsTr(keep);
data.labelsTr = data.labelsTr(keep);

%% Set nans to zero (this should not happen)
data.attReprTe(isnan(data.attReprTe))=0;
data.attReprTr(isnan(data.attReprTr))=0;
data.phocsTe(isnan(data.phocsTe))=0;
data.phocsTr(isnan(data.phocsTr))=0;

%% Randomize train
p = randperm(size(data.attReprTr,2));
data.attReprTr = data.attReprTr(:,p);
data.phocsTr = data.phocsTr(:,p);
data.wordClsTr = data.wordClsTr(p);
data.labelsTr = data.labelsTr(p);
data.TestPermutation = p;

%% Create "fake" validation partition
% Extract 30% descs from train to use as validation.
% We also keep the original full training set to retrain later on.

data.attReprTrFull = data.attReprTr;
data.phocsTrFull = data.phocsTr;
data.wordClsTrFull = data.wordClsTr;
data.labelsTrFull = data.labelsTr;

nval = floor(0.3*size(data.attReprTr,2));
data.attReprVa = data.attReprTr(:,1:nval);
data.attReprTr = data.attReprTr(:,nval+1:end);
data.phocsVa = data.phocsTr(:,1:nval);
data.phocsTr = data.phocsTr(:,nval+1:end);
data.wordClsVa = data.wordClsTr(1:nval);
data.wordClsTr = data.wordClsTr(nval+1:end);
data.labelsVa = data.labelsTr(1:nval);
data.labelsTr = data.labelsTr(nval+1:end);

embedding_final = [];
embRepr_final = [];

mAPdirect = evaluateDirect(opts, data);
bestmAP = mAPdirect;

mAPplatts = 0;
mAPreg = 0;
mAPcca = 0;
mAPkcca = 0;

if opts.TestPlatts
    [embedding,emb_repr,mAPplatts] = evaluatePlatts(opts, data);
    if mAPplatts > bestmAP
        embedding_final = embedding;
        embRepr_final = emb_repr;
        bestmAP = mAPplatts;
    end
end

if opts.TestRegress
    [embedding,emb_repr,mAPreg]= evaluateReg(opts, data);
    if mAPreg > bestmAP
        embedding_final = embedding;
        embRepr_final = emb_repr;
        bestmAP = mAPreg;
    end
end

if opts.TestCCA
    [embedding,emb_repr,mAPcca] = evaluateCCA(opts, data);
    if mAPcca > bestmAP
        embedding_final = embedding;
        embRepr_final = emb_repr;
        bestmAP = mAPcca;
    end
end

if opts.TestKCCA
    [embedding,emb_repr,mAPkcca] = evaluateKCCA(opts, data);
    if mAPkcca > bestmAP
        embedding_final = embedding;
        embRepr_final = emb_repr;
        bestmAP = mAPkcca;
    end
end

if opts.evalLineSpotting
    load('IAM_dict_line2classes.mat');
    data.line2classes = line2classes;
end
if opts.evalRecog
    lexicon = load(opts.lexiconFile);
    data.lexicon = lexicon;
end




if opts.evalLineSpotting
    % Preparing queries for Volkmar's protocol
    DATA.wordsTr = DATA.wordsTr';
    DATA.wordsVal = DATA.wordsVal';
    
    words = [DATA.wordsTr DATA.wordsVal DATA.wordsTe];
    phocs = [DATA.phocsTr DATA.phocsVal DATA.phocsTe];
    queriesClasses = [DATA.queriesClassesTr; DATA.queriesClassesVal; DATA.queriesClassesTe];
    [u,ind,n] = unique(lower(words(:)));
    B = accumarray(n, 1, [], @sum);
    [B,I] = sort(B,'descend');
    words = u(I);
    ind = ind(I);
    [words,ia,ib] = intersect(upper(words),[DATA.wordsTr DATA.wordsVal],'stable');
    words = words(1:4000);
    ind = ind(ia);
    ind = ind(1:4000);
    phocs = phocs(:,ind);
    queriesClasses = queriesClasses(ind);
    tmp = mat*phocs;
    phocs_cca = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
    phocs_cca=bsxfun(@minus, phocs_cca, mh);
    phocs_cca = Wy(:,1:bestK)' * phocs_cca;
    phocs_cca = (bsxfun(@rdivide, phocs_cca, sqrt(sum(phocs_cca.*phocs_cca))));
    
    mapLineSpotting = eval_line_spotting_Volkmar(opts, phocs_cca, attReprTe_cca,queriesClasses,DATA.queriesClassesTe,words,DATA.linesTe,DATA.line2classes);
    
    % mapLineSpotting = eval_line_spotting(opts, phocs_cca, attReprTe_cca,DATA.queriesClassesTe,DATA.wordsTe,DATA.linesTe,DATA.line2classes,1);
    mean(mapLineSpotting)
end



if opts.evalRecog
    phocs = single(DATA.lexicon.phocs);
    tmp = mat*phocs;
    phocs_cca = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
    phocs_cca=bsxfun(@minus, phocs_cca, mh);
    phocs_cca = Wy(:,1:bestK)' * phocs_cca;
    phocs_cca = (bsxfun(@rdivide, phocs_cca, sqrt(sum(phocs_cca.*phocs_cca))));
    phocs_cca(isnan(phocs_cca)) = 0;
    words = DATA.lexicon.words;
    
    N = size(attReprTe_cca,2);
    p1small = zeros(N,1);
    p1medium = zeros(N,1);
    %     load('SVT_IIIT5K_images.mat', 'svt');
    %     iiit5kRecog = iiit5k;
    %     svtRecog = svt;
    for i=1:N
        feat = attReprTe_cca(:,i);
        gt = DATA.lexicon.testdata(i).GroundTruth;
        
        smallLexi = DATA.lexicon.testdata(i).smallLexi;
        [~,~,ind] = intersect(smallLexi,words,'stable');
        scores = feat'*phocs_cca(:,ind);
        randInd = randperm(length(scores));
        scores = scores(randInd);
        [scores,I] = sort(scores,'descend');
        I = randInd(I);
        
        
        if strcmpi(gt,smallLexi{I(1)})
            p1small(i) = 1;
        else
            p1small(i) = 0;
        end
        
        %         svtRecog(i).recogSmall = smallLexi{I(1)};
        
        %         mediumLexi = DATA.lexicon.testdata(i).mediumLexi;
        %         [~,~,ind] = intersect(mediumLexi,words,'stable');
        %         scores = feat'*phocs_cca(:,ind);
        %         randInd = randperm(length(scores));
        %         scores = scores(randInd);
        %         [scores,I] = sort(scores,'descend');
        %         I = randInd(I);
        %         if strcmpi(gt,mediumLexi(I(1)))
        %             p1medium(i) = 1;
        %         else
        %             p1medium(i) = 0;
        %         end
        %
        %         iiit5kRecog(i).recogMedium = mediumLexi{I(1)};
        
    end
    disp('------------------------------------');
    fprintf('lexicon small --   map: %.2f\n', 100*mean(p1small));
    fprintf('lexicon medium --   map: %.2f\n', 100*mean(p1medium));
    disp('------------------------------------');
end

% Hybrid (qbe style)
if opts.evalHybrid
alpha = 0:0.1:1;
hybrid_test_map = zeros(length(alpha),1);
for i=1:length(alpha)
    attRepr_hybrid = attReprTe_cca*alpha(i) + phocsTe_cca*(1-alpha(i));
    [p1,mAPEucl,q] = eval_dp_asymm(opts,attRepr_hybrid, attReprTe_cca,DATA.queriesClassesTe,DATA.wordsTe);
    hybrid_test_map(i) = mean(mAPEucl);
end
end

end