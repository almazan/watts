function phocs = embed_labels_PHOC(opts,data)

if ~exist(opts.filePHOCs,'file')
    nWords = length(data.words);
    
    [drop,dhoc] = compute_phoc('null',opts.levels,opts.considerDigits);
    phocs = zeros(dhoc,nWords,'single');
    parfor i=1:nWords
        phocs(:,i) = compute_phoc(data.words(i).gttext,opts.levels,opts.considerDigits);
    end
    
    [drop,dhob] = compute_phob('null',opts.bgrams,opts.levelsB);
    phobs = zeros(dhob,nWords,'single');
    parfor i=1:nWords
        phobs(:,i) = compute_phob(data.words(i).gttext,opts.bgrams,opts.levelsB);
    end
    
    phocs = [phocs; phobs];
    
    save(opts.filePHOCs,'phocs');
else
    load(opts.filePHOCs);
end

end