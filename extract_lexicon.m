function lexicon = extract_lexicon(opts,data)
words = data.words(data.idxTest);

% Extracts the unique set of words in the lexicon
if ismember(opts.dataset,{'IIIT5K','ICDAR11'})
    words=[words(:).sLexi words(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'SVT')
    words=[words(:).sLexi];
    words = unique(words);
end

% Extracts the PHOC embedding for every word in the lexicon
nWords = length(words);

% PHOC of single characters using the number of levels in opts.levels
[drop,dhoc] = compute_phoc('null',opts.levels,opts.considerDigits);
phocs = zeros(dhoc,nWords,'single');
parfor i=1:nWords
    phocs(:,i) = compute_phoc(words{i},opts.levels,opts.considerDigits);
end

% PHOB of the N most common bigrams specified in opts.bgrams using the
% number of levels in opts.leves
[drop,dhob] = compute_phob('null',opts.bgrams,opts.levelsB);
phobs = zeros(dhob,nWords,'single');
parfor i=1:nWords
    phobs(:,i) = compute_phob(words{i},opts.bgrams,opts.levelsB);
end

% Histograms of characters and bigrams are concatenated in the final
% PHOC representation
phocs = [phocs; phobs];


lexicon.words = words;
lexicon.phocs = phocs;

save(opts.fileLexicon,'lexicon');

end