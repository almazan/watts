function recognition = evaluate_recognition(opts,data,embedding)

% Small fix for versions of matlab older than 2012b ('8') that do not support stable intersection
if verLessThan('matlab', '8')
    inters=@stableintersection;
else
    inters=@intersect;
end

emb = embedding.kcca;

% Create/load the dictionary (lexicon)
if ~exist(opts.fileLexicon,'file')
    lexicon = extract_lexicon(opts,data);
else
    load(opts.fileLexicon,'lexicon')
end

%lexicon.phocs = [lexicon.phocs;encodeWordsLength(lexicon.words,10)];

% Embed the test representations
matx = emb.rndmatx(1:emb.M,:);
maty = emb.rndmaty(1:emb.M,:);
tmp = matx*data.attReprTe;
attReprTe_emb = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
tmp = maty*data.phocsTe;
phocsTe_emb = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
attReprTe_emb=bsxfun(@minus, attReprTe_emb, emb.matts);
phocsTe_emb=bsxfun(@minus, phocsTe_emb, emb.mphocs);
attReprTe_emb = emb.Wx(:,1:emb.K)' * attReprTe_emb;
phocsTe_emb = emb.Wy(:,1:emb.K)' * phocsTe_emb;
attReprTe_emb = (bsxfun(@rdivide, attReprTe_emb, sqrt(sum(attReprTe_emb.*attReprTe_emb))));
phocsTe_emb = (bsxfun(@rdivide, phocsTe_emb, sqrt(sum(phocsTe_emb.*phocsTe_emb))));

% Embed the dictionary
phocs = single(lexicon.phocs);
maty = emb.rndmaty(1:emb.M,:);
tmp = maty*phocs;
phocs_cca = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
phocs_cca=bsxfun(@minus, phocs_cca, emb.mphocs);
phocs_cca = emb.Wy(:,1:emb.K)' * phocs_cca;
phocs_cca = (bsxfun(@rdivide, phocs_cca, sqrt(sum(phocs_cca.*phocs_cca))));
phocs_cca(isnan(phocs_cca)) = 0;
words = lexicon.words;

if strcmpi(opts.dataset,'IAM')
    [cer, p1, qidx] = compute_cer(attReprTe_emb,phocs_cca,data.wordClsTe, data.labelsTe');        
    linesTe = {data.wordsTe.lineId}';
    linesTe = linesTe(qidx);
    wer = compute_wer(linesTe,p1);
    disp(cer);
    disp(wer);
    
    
else
    N = size(attReprTe_emb,2);
    p1small = zeros(N,1);
    p1medium = zeros(N,1);
    p1large = zeros(N,1);
    for i=1:N
        feat = attReprTe_emb(:,i);
        gt = data.wordsTe(i).gttext;
        if ~strcmpi(opts.dataset, 'LP')
            smallLexicon = unique(data.wordsTe(i).sLexi);
            [~,~,ind] = inters(smallLexicon,words,'stable');
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
        end
        if strcmpi(opts.dataset,'IIIT5K')
            mediumLexicon = unique(data.wordsTe(i).mLexi);
            [~,~,ind] = inters(mediumLexicon,words,'stable');
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
        
        scores = feat'*phocs_cca;
        randInd = randperm(length(scores));
        scores = scores(randInd);
        [scores,I] = sort(scores,'descend');
        I = randInd(I);
        
        if strcmpi(gt,words{I(1)})
            p1large(i) = 1;
        else
            p1large(i) = 0;
        end
        
    end
    disp('------------------------------------');
    recognition.small = 100*mean(p1small);
    fprintf('lexicon small --   p@1: %.2f\n', recognition.small);
    if strcmpi(opts.dataset,'IIIT5K')
        recognition.medium = 100*mean(p1medium);
        fprintf('lexicon medium --   p@1: %.2f\n', recognition.medium);
    end
    recognition.large = 100*mean(p1large);
    fprintf('lexicon large --   p@1: %.2f\n', recognition.large);
    disp('------------------------------------');
end
end

% Ugly hack to deal with the lack of stable intersection in old versions of
% matlab
function [empty1, empty2, ind] = stableintersection(a, b, varargin)
empty1=0;
empty2=0;
[~,ia,ib] = intersect(a,b);
[~, tmp2] = sort(ia);
ind = ib(tmp2);
end
