function phocs = embed_labels_PHOC(opts,data)
% Computes the PHOC embedding for every word label in the dataset

if  ~exist(opts.filePHOCs,'file')
    voc = opts.unigrams;
    if opts.considerDigits
        voc = [voc opts.digits];
    end
    str2cell = @(x) {char(x)};
    voc = arrayfun(str2cell, voc);
    
    lf = @(x) lower(x.gttext);
    W = arrayfun(lf, data.words,'UniformOutput', false);
    
    
    phocsuni = phoc_mex(W, voc, int32(opts.levels));
    phocsbi = phoc_mex(W, opts.bgrams, int32(opts.levelsB));
    phocs = [phocsuni;phocsbi];   
    
    save(opts.filePHOCs,'phocs', '-v7.3');
else
    load(opts.filePHOCs);
end

end
