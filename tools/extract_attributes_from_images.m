function attRepr = extract_attributes_from_images(opts,images,attModels,embedding)

try
    load(opts.fileGMM);
    load(opts.filePCA);
catch
    disp('Please, compute GMM and PCA models first.');
    return;
end
images = {images(:).im};
attRepr = extract_FV_features(opts,images,GMM,PCA);
W = [attModels(:).W];

attRepr = W'*attRepr;

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

end