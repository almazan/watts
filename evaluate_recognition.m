function recognition = evaluate_recognition(opts,data,embedding)

% Small fix for versions of matlab older than 2012b ('8') that do not support stable intersection
if verLessThan('matlab', '8')
    inters=@stableintersection;
else
    inters=@intersect;
end


% Create/load the dictionary (lexicon)
if ~exist(opts.fileLexicon,'file')
    lexicon = extract_lexicon(opts,data);
else
    load(opts.fileLexicon,'lexicon')
end

lexicon_phocs = single(lexicon.phocs);
words = lexicon.words;

if opts.TestCCA
    emb = embedding.cca;
    % Embed the test representations
    attReprTe_emb = bsxfun(@rdivide, data.attReprTe,sqrt(sum(data.attReprTe.*data.attReprTe)));
    attReprTe_emb(isnan(attReprTe_emb)) = 0;
    attReprTe_emb =  bsxfun(@minus, attReprTe_emb,emb.matts);
    attReprTe_emb = emb.Wx(:,1:emb.K)' * attReprTe_emb;
    attReprTe_emb = (bsxfun(@rdivide, attReprTe_emb, sqrt(sum(attReprTe_emb.*attReprTe_emb))));
    
    % Embed the lexicon dictionary
    lexicon_phocs_emb = bsxfun(@rdivide, lexicon_phocs,sqrt(sum(lexicon_phocs.*lexicon_phocs)));
    lexicon_phocs_emb=  bsxfun(@minus, lexicon_phocs_emb,emb.mphocs);
    lexicon_phocs_emb = emb.Wy(:,1:emb.K)' * lexicon_phocs_emb;
    lexicon_phocs_emb = (bsxfun(@rdivide, lexicon_phocs_emb, sqrt(sum(lexicon_phocs_emb.*lexicon_phocs_emb))));
    
elseif opts.TestKCCA
    emb = embedding.kcca;
    % Embed the test representations
    matx = emb.rndmatx(1:emb.M,:);
    tmp = matx*data.attReprTe;
    attReprTe_emb = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
    attReprTe_emb=bsxfun(@minus, attReprTe_emb, emb.matts);
    attReprTe_emb = emb.Wx(:,1:emb.K)' * attReprTe_emb;
    attReprTe_emb = (bsxfun(@rdivide, attReprTe_emb, sqrt(sum(attReprTe_emb.*attReprTe_emb))));
    
    % Embed the lexicon dictionary
    maty = emb.rndmaty(1:emb.M,:);
    tmp = maty*lexicon_phocs;
    lexicon_phocs_emb = 1/sqrt(emb.M) * [ cos(tmp); sin(tmp)];
    lexicon_phocs_emb=bsxfun(@minus, lexicon_phocs_emb, emb.mphocs);
    lexicon_phocs_emb = emb.Wy(:,1:emb.K)' * lexicon_phocs_emb;
    lexicon_phocs_emb = (bsxfun(@rdivide, lexicon_phocs_emb, sqrt(sum(lexicon_phocs_emb.*lexicon_phocs_emb))));
    lexicon_phocs_emb(isnan(lexicon_phocs_emb)) = 0;
end


% Get all the valid queries. For most datasets, that is all of them.
qidx = find(~strcmp({data.wordsTe.gttext},'-'));
N = length(qidx);

% Get the scores: p1, cer, and wer
p1small = zeros(N,1);
cersmall = zeros(N,1);
wersmall = zeros(N,1);
p1medium = zeros(N,1);
cermedium = zeros(N,1);
wermedium = zeros(N,1);
p1full = zeros(N,1);
cerfull = zeros(N,1);
werfull = zeros(N,1);

for i=1:N
    % Get actual idx, feature vector, and gt transcription
    pos = qidx(i);
    feat = attReprTe_emb(:,pos);
    gt = data.wordsTe(pos).gttext;
    
    % Small lexicon available
    if isfield(data.wordsTe,'sLexi')
        smallLexicon = unique(data.wordsTe(i).sLexi);
        [~,~,ind] = inters(smallLexicon,words,'stable');
        scores = feat'*lexicon_phocs_emb(:,ind);
        randInd = randperm(length(scores));
        scores = scores(randInd);
        [scores,I] = sort(scores,'descend');
        I = randInd(I);
        
        if strcmpi(gt,smallLexicon{I(1)})
            p1small(i) = 1;
        else
            p1small(i) = 0;
            cersmall(i) = levenshtein_c(gt, smallLexicon{I(1)});
        end
    end
    
    % Medium lexicon available
    if isfield(data.wordsTe,'mLexi')
        mediumLexicon = unique(data.wordsTe(pos).mLexi);
        [~,~,ind] = inters(mediumLexicon,words,'stable');
        scores = feat'*lexicon_phocs_emb(:,ind);
        randInd = randperm(length(scores));
        scores = scores(randInd);
        [scores,I] = sort(scores,'descend');
        I = randInd(I);
        if strcmpi(gt,mediumLexicon(I(1)))
            p1medium(i) = 1;
        else
            p1medium(i) = 0;
            cermedium(i) = levenshtein_c(gt, mediumLexicon{I(1)});
        end
    end
    
    % Full lexicon always available
    scores = feat'*lexicon_phocs_emb;
    randInd = randperm(length(scores));
    scores = scores(randInd);
    [scores,I] = sort(scores,'descend');
    I = randInd(I);
    
    if strcmpi(gt,words{I(1)})
        p1full(i) = 1;
    else
        p1full(i) = 0;
        cerfull(i) = levenshtein_c(gt, words{I(1)});
    end
end

% Compute wer if there is line info available
if isfield(data.wordsTe,'lineId')
    linesTe = {data.wordsTe.lineId}';
    linesTe = linesTe(qidx);
    if isfield(data.wordsTe,'sLexi')
        recognition.wersmall = compute_wer(linesTe,p1small);
    end
    if isfield(data.wordsTe,'mLexi')
        recognition.wermedium = compute_wer(linesTe,p1medium);
    end
    recognition.werfull = compute_wer(linesTe,p1full);
end


% Display stuff
fprintf('\n');
disp('**************************************');
disp('************  Recognition  ***********');
disp('**************************************');
disp('------------------------------------');
if isfield(data.wordsTe,'sLexi')
    recognition.p1small = 100*mean(p1small);
    recognition.cersmall = 100*mean(cersmall);
    if isfield(recognition,'wersmall')
        fprintf('lexicon small  -- p@1: %.2f. cer: %.2f. wer: %.2f\n', recognition.p1small, recognition.cersmall, recognition.wersmall);
    else
        fprintf('lexicon small  -- p@1: %.2f. cer: %.2f. wer: N/A\n', recognition.p1small, recognition.cersmall);
    end
end

if isfield(data.wordsTe,'mLexi')
    recognition.p1medium = 100*mean(p1medium);
    recognition.cermedium = 100*mean(cermedium);
    if isfield(recognition,'wermedium')
        fprintf('lexicon medium -- p@1: %.2f. cer: %.2f. wer: %.2f\n', recognition.p1medium, recognition.cermedium, recognition.wermedium);
    else
        fprintf('lexicon medium -- p@1: %.2f. cer: %.2f. wer: N/A\n', recognition.p1medium, recognition.cermedium);
    end
end

recognition.p1full = 100*mean(p1full);
recognition.cerfull = 100*mean(cerfull);
if isfield(recognition,'werfull')
    fprintf('lexicon full   -- p@1: %.2f. cer: %.2f. wer: %.2f\n', recognition.p1full, recognition.cerfull, recognition.werfull);
else
    fprintf('lexicon full   -- p@1: %.2f. cer: %.2f. wer: N/A\n', recognition.p1full, recognition.cerfull);
end
disp('------------------------------------');
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
