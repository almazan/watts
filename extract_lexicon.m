function lexicon = extract_lexicon(opts,data)
words = data.words(data.idxTest);

% Extracts the unique set of words in the lexicon
if strcmpi(opts.dataset,'IIIT5K')
    words=[words(:).sLexi words(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'ICDAR11')
    words=[words(:).sLexi words(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'SVT')
    words=[words(:).sLexi];
    words = unique(words);
end

% Extracts the PHOC embedding for every word in the lexicon
voc = opts.unigrams;
if opts.considerDigits
    voc = [voc opts.digits];
end
str2cell = @(x) {char(x)};
voc = arrayfun(str2cell, voc);

lf = @(x) lower(x);
W = cellfun(lf, words,'UniformOutput', false);

phocsuni = phoc_mex(W, voc, int32(opts.levels));
phocsbi = phoc_mex(W, opts.bgrams, int32(opts.levelsB));
phocs = [phocsuni;phocsbi];

lexicon.words = words;
lexicon.phocs = phocs;

save(opts.fileLexicon,'lexicon');

end