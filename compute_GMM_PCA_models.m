function [GMM,PCA] = compute_GMM_PCA_models(opts,images)

descrs = {};
for i=1:length(images)
    fprintf('Word %d\n', i);
    im = images{i};
    
    % Resizes the image to a minimum height without modifying the aspect
    % ratio
    [height,width] = size(im);
    if height<opts.minH
        ar = height/width;
        height = opts.minH;
        width = round(height/ar);
        im = imresize(im, [height,width]);
    end
    
    im = im2single(im);
    
    % Densely extracts SIFTs at different levels
    [f,d] = vl_phow(im, opts.phowOpts{:});
    d = d/255;
    
    
    if opts.doMinibox == 0
        % XY at GT coordinate space
        fx = single(f(1,:)/width-0.5);
        fy = single(f(2,:)/height-0.5);
    else
        % XY at word coordinate space
        bb = DoBB(im);
        w = bb(2)-bb(1)+1;
        h = bb(4)-bb(3)+1;
        cx = round(bb(1)+w/2);
        cy = round(bb(3)+h/2);
        fx = single((f(1,:)-cx)/w);
        fy = single((f(2,:)-cy)/h);
    end
    xy = [fx; fy];
    d = [d; xy];
    
    % Assings each SIFT to a region of the spatial pyramid
    for s = 1:length(opts.numSpatialX)
        ax = linspace(-0.5,0.5,opts.numSpatialX(s)+1);
        ax(1) = -Inf; ax(end) = Inf;
        ay = [-Inf 0 Inf];
        binsx = vl_binsearch(ax,double(fx));
        binsy = vl_binsearch(ay,double(fy));
        
        for j=1:opts.numSpatialX(s)
            for k=1:opts.numSpatialY(s)
                idx = (binsx==j) & (binsy==k);
                descrs{s}{j,k}{i} = d(:,idx);
            end
        end
        
    end
end

%% Computing global PCA
% Selects a subset of normalized SIFTs
disp('* Computing PCA model *');
d = [descrs{:}];
d = [d{:}];
d = [d{:}];
[d,sel] = vl_colsubset(d, 20e5);
[d,drop] = normalizeSift(opts,d);
[eigvec, m] = compute_PCA(d(1:opts.SIFTDIM,:),opts.PCADIM);
PCA.eigvec = eigvec;
PCA.mean = m;


%% Computing GMM
% Computing a GMM for every region of the spatial pyramid and concatenate
disp('* Computing GMM model *');
GMM.we =[];
GMM.mu = [];
GMM.sigma = [];
for s = 1:length(opts.numSpatialX)
    for j=1:opts.numSpatialX(s)
        for k=1:opts.numSpatialY(s)
            d = cat(2, descrs{s}{j,k}{:});
            [d,drop] = normalizeSift(opts,d);
            xy = d(opts.SIFTDIM+1:end,:);
            d=bsxfun(@minus, d(1:opts.SIFTDIM,:), PCA.mean);
            d=PCA.eigvec'*d;
            
            d = [d; xy];
            [mu,sigma,we] = vl_gmm(d, opts.G, 'MaxNumIterations', 30, 'NumRepetitions', 2); 
            we = we'; 
            GMM.we = [GMM.we we];
            GMM.mu = [GMM.mu mu];
            GMM.sigma = [GMM.sigma sigma];
        end
    end
end
GMM.we = GMM.we/sum(GMM.we);



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

function [eigvec, m] = compute_PCA(X,PCADIM)
m = mean(X,2);
[eigvec,eigval]=eig(cov(X'));
[a,I] =  sort(diag(eigval), 'descend');
eigvec=eigvec(:,I(1:PCADIM));