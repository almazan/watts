function phocs = embed_labels_PHOC(opts,data)
% Computes the PHOC embedding for every word label in the dataset

if ~exist(opts.filePHOCs,'file')
    nWords = length(data.words);
    
    % PHOC of single characters using the number of levels in opts.levels
    [drop,dhoc] = compute_phoc('null',opts.levels,opts.considerDigits);
    phocs = zeros(dhoc,nWords,'single');
    parfor i=1:nWords
        phocs(:,i) = compute_phoc(data.words(i).gttext,opts.levels,opts.considerDigits);
    end
    
    % PHOB of the N most common bigrams specified in opts.bgrams using the
    % number of levels in opts.leves
    [drop,dhob] = compute_phob('null',opts.bgrams,opts.levelsB);
    phobs = zeros(dhob,nWords,'single');
    parfor i=1:nWords
        phobs(:,i) = compute_phob(data.words(i).gttext,opts.bgrams,opts.levelsB);
    end
    
    % Histograms of characters and bigrams are concatenated in the final
    % PHOC representation
    phocs = [phocs; phobs];
    
    save(opts.filePHOCs,'phocs');
else
    load(opts.filePHOCs);
end

end