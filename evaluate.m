function mAP = evaluate(opts,data,embedding)

%% Load attribute representations
load(opts.fileAttRepres,'attReprTe');

data.attReprTe = single(attReprTe);
data.phocsTe = single(data.phocsTe);
data.wordClsTe = single(data.wordClsTe);

mAP.fv = [];
mAP.direct = [];
mAP.platts = [];
mAP.reg = [];
mAP.cca = [];
mAP.kcca = [];

if opts.TestFV
    mAP.fv = evaluateFV(opts, data);
end

if opts.TestDirect
    mAP.direct = evaluateDirect(opts, data);
end

if opts.TestPlatts
    mAP.platts = evaluatePlatts(opts, data, embedding.platts);
end

if opts.TestRegress
    mAP.reg = evaluateReg(opts, data, embedding.reg);
end

if opts.TestCCA
    mAP.cca = evaluateCCA(opts, data, embedding.cca);
end

if opts.TestKCCA
    mAP.kcca = evaluateKCCA(opts, data, embedding.kcca);
end

if opts.evalRecog
    emb = embedding.kcca;
    
    % Create/load the dictionary (lexicon)
    if ~exist(opts.fileLexicon,'file')
        lexicon = extract_lexicon(opts,data);
    else
        load(opts.fileLexicon,'lexicon')
    end
    
    % Embed the test attributes representation (attRepreTe_emb)
    mat = emb.rndmat(1:emb.M,:);
    tmp = mat*data.attReprTe;
    attReprTe_emb = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
    attReprTe_emb = bsxfun(@minus, attReprTe_emb, emb.matts);
    attReprTe_emb = emb.Wx(:,1:emb.K)' * attReprTe_emb;
    attReprTe_emb = (bsxfun(@rdivide, attReprTe_emb, sqrt(sum(attReprTe_emb.*attReprTe_emb))));
    
    % Embed the dictionary
    phocs = single(lexicon.phocs);
    tmp = mat*phocs;
    phocs_cca = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
    phocs_cca=bsxfun(@minus, phocs_cca, emb.mphocs);
    phocs_cca = emb.Wy(:,1:emb.K)' * phocs_cca;
    phocs_cca = (bsxfun(@rdivide, phocs_cca, sqrt(sum(phocs_cca.*phocs_cca))));
    phocs_cca(isnan(phocs_cca)) = 0;
    words = lexicon.words;
    
    N = size(attReprTe_emb,2);
    p1small = zeros(N,1);
    p1medium = zeros(N,1);
    for i=1:N
        feat = attReprTe_emb(:,i);
        gt = data.wordsTe(i).gttext;
        
        smallLexicon = data.wordsTe(i).sLexi;
        [~,~,ind] = intersect(smallLexicon,words,'stable');
        scores = feat'*phocs_cca(:,ind);
        randInd = randperm(length(scores));
        scores = scores(randInd);
        [scores,I] = sort(scores,'descend');
        I = randInd(I);
        
        if strcmpi(gt,smallLexicon{I(1)})
            p1small(i) = 1;
        else
            p1small(i) = 0;
        end
        
        if strcmpi(opts.dataset,'IIIT5K')
            mediumLexicon = data.wordsTe(i).mLexi;
            [~,~,ind] = intersect(mediumLexicon,words,'stable');
            scores = feat'*phocs_cca(:,ind);
            randInd = randperm(length(scores));
            scores = scores(randInd);
            [scores,I] = sort(scores,'descend');
            I = randInd(I);
            if strcmpi(gt,mediumLexicon(I(1)))
                p1medium(i) = 1;
            else
                p1medium(i) = 0;
            end
        end
    end
    disp('------------------------------------');
    fprintf('lexicon small --   map: %.2f\n', 100*mean(p1small));
    if strcmpi(opts.dataset,'IIIT5K')
        fprintf('lexicon medium --   map: %.2f\n', 100*mean(p1medium));
    end
    disp('------------------------------------');
end

%
% % Hybrid (qbe style)
% if opts.evalHybrid
%     alpha = 0:0.1:1;
%     hybrid_test_map = zeros(length(alpha),1);
%     for i=1:length(alpha)
%         attRepr_hybrid = attReprTe_cca*alpha(i) + phocsTe_cca*(1-alpha(i));
%         [p1,mAPEucl,q] = eval_dp_asymm(opts,attRepr_hybrid, attReprTe_cca,DATA.queriesClassesTe,DATA.wordsTe);
%         hybrid_test_map(i) = mean(mAPEucl);
%     end
% end

end