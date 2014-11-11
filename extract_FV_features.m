function feats = extract_FV_features(opts,images,GMM,PCA)

nWords = length(images);
feats = zeros(opts.FVdim,nWords,'single');

parfor i=1:length(images)
    fprintf('Extracting FV representation from image %d\n',i);
    
    im = images{i};
    [height,width] = size(im);
    im = im2single(im);
    
    % get PHOW features
    descrs = [];
    frames = [];
    if height>1 && width>1
        [frames, descrs] = vl_phow(im, opts.phowOpts{:}) ;
        descrs = descrs / 255;
    end
    
    if isempty(descrs)
        continue;
    end
    
    if opts.doMinibox == 0
        % XY at GT coordinate space
        fx = single(frames(1,:)/width-0.5);
        fy = single(frames(2,:)/height-0.5);
    else
        % XY at word coordinate space
        bb = DoBB(im);
        w = bb(2)-bb(1)+1;
        h = bb(4)-bb(3)+1;
        cx = round(bb(1)+w/2);
        cy = round(bb(3)+h/2);
        fx = single((frames(1,:)-cx)/w);
        fy = single((frames(2,:)-cy)/h);
    end
    xy = [fx; fy];
    descrs = [descrs; xy];
    
    
    [descrs,frames] = normalizeSift(opts,descrs,frames);
    
    feats(:,i) = single(getImageDescriptorFV(opts, GMM, PCA, descrs));
    
end
feats(isnan(feats)) = 0;
end

% -------------------------------------------------------------------------
function fv = getImageDescriptorFV(opts, GMM, PCA, descrs)
% -------------------------------------------------------------------------

% Project into PCA space
xy = descrs(opts.SIFTDIM+1:end,:);
descrs=bsxfun(@minus, descrs(1:opts.SIFTDIM,:), PCA.mean);
descrs=PCA.eigvec'*descrs;

descrs = [descrs; xy];

% Extracts FV representation using the GMM
fv  =  vl_fisher(descrs, GMM.mu, GMM.sigma, GMM.we, 'Improved');
end

function X = normFV(X)
% -------------------------------------------------------------------------
X = sign(X).*sqrt(abs(X));
X = bsxfun(@rdivide, X, sqrt(sum(X.*X)));
X(isnan(X)) = 0;
end

function [descrs_normalized,frames_normalized] = normalizeSift(opts,descrs,frames)
% -------------------------------------------------------------------------
descrs_normalized = descrs;

xy = descrs_normalized(opts.SIFTDIM+1:end,:);
descrs_normalized = descrs_normalized(1:opts.SIFTDIM,:);

% Remove empty ones
idx = find(sum(descrs_normalized)==0);
descrs_normalized(:,idx)=[];
if nargin < 3
    frames_normalized = [];
else
    frames_normalized = frames;
    frames_normalized(:,idx) = [];
end

% Square root:
descrs_normalized = sqrt(descrs_normalized);

% 1/4 norm
X = sum(descrs_normalized.*descrs_normalized).^-0.25;
descrs_normalized = bsxfun(@times, descrs_normalized,X);

xy(:,idx) = [];
descrs_normalized = [descrs_normalized; xy];

descrs_normalized(isnan(descrs_normalized))=0;
end

function im = adjustImage(im)
imOrig = im;
im = im2bw(im);
[h,w] = size(im);
x = find(im==0);
w1 = ceil(min(x)/h);
w2 = floor(max(x)/h);
h1 = min(mod(x,h))+1;
h2 = max(mod(x,h))-1;
im = imOrig(h1:h2,w1:w2);
end
