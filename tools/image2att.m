function attRepr = image2att(images,folderData,embeddingMethod)

addpath('../');
addpath('../util/');
addpath('../util/io/');

if nargin<3
    embeddingMethod = 'cca';
end

if ~exist(folderData,'dir')
    error('Please, compute PCA, GMM and attribute models first');
end

% Loading data
load(fullfile(folderData,'opts.mat'));
PCA = readPCA(fullfile(folderData,'PCA.bin'));
GMM = readGMM(fullfile(folderData,'GMM.bin'));
if strcmpi(embeddingMethod,'cca')
    embedding = readCCA(fullfile(folderData,'CCA.bin'));
elseif strcmpi(embeddingMethod,'kcca')
    embedding = readKCCA(fullfile(folderData,'KCCA.bin'));
else
    embedding = struct();
end
attModels = readMat(fullfile(folderData,'attModels.bin'));

if ~iscell(images)
    images = {images};
end

% Preparing images
for i=1:length(images)
    im = images{i};
    [H,W,numC] = size(im);
    if numC > 1
        im = rgb2gray(im);
    end
    % Move to single and equalize if necessary
    im = im2single(im);
    m = max(max(im));
    if m < 0.2
        im = im*0.2/m;
    end
    
    if  (opts.minH > H)
        im = imresize(im, [opts.minH,nan]);
    end
    if  (opts.maxH < H)
        im = imresize(im, [opts.maxH,nan]);
    end
    images{i} = im;
end

% Project to attribute space
attRepr = extract_FV_features(opts,images,GMM,PCA);
W = attModels(1:end-1,:);
attRepr = W'*attRepr;

% Project to CCA/KCCA space if necessary
if isfield(embedding,'rndmat') % KCCA
    mat = embedding.rndmat(1:embedding.M,:);
    
    tmp = mat*attRepr;
    attRepr = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
    
    % Mean center
    attRepr=bsxfun(@minus, attRepr, embedding.matts);
    
    % Embed test
    attRepr = embedding.Wx(:,1:embedding.K)' * attRepr;
    
    % L2 normalize (critical)
    attRepr = (bsxfun(@rdivide, attRepr, sqrt(sum(attRepr.*attRepr))));
    attRepr(isnan(attRepr)) = 0;
elseif isfield(embedding,'K')% CCA
    % L2 normalize
    attRepr = bsxfun(@rdivide, attRepr,sqrt(sum(attRepr.*attRepr)));
    attRepr(isnan(attRepr)) = 0;
    
    attRepr =  bsxfun(@minus, attRepr,embedding.matts);
    
    % Embed  test
    attRepr = embedding.Wx(:,1:embedding.K)' * attRepr;
    
    % L2 normalize (critical)
    attRepr = (bsxfun(@rdivide, attRepr, sqrt(sum(attRepr.*attRepr))));
end

end