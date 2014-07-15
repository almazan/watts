function lexicon = extract_lexicon(opts,data)
% Small fix for versions of matlab older than 2012b ('8') that do not support stable intersection
if verLessThan('matlab', '8')
    inters=@stableintersection;
else
    inters=@intersect;
end

% Extracts the unique set of words in the lexicon
if strcmpi(opts.dataset,'IIIT5K')
    wordsTe = data.wordsTe;
    words=[wordsTe(:).sLexi wordsTe(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'ICDAR11')
    wordsTe = data.wordsTe;
    words=[wordsTe(:).sLexi wordsTe(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'ICDAR03')
    wordsTe = data.wordsTe;
    words=[wordsTe(:).sLexi wordsTe(:).mLexi];
    words = unique(words);
elseif strcmpi(opts.dataset,'SVT')
    wordsTe = data.wordsTe;
    words=[wordsTe(:).sLexi];
    words = unique(words);
elseif strcmpi(opts.dataset, 'LP')
    wordsTe = data.words;
    words = unique({wordsTe.gttext})';
elseif strcmpi(opts.dataset, 'IAM')
    wordsTe = data.wordsTe;
    words = unique({wordsTe.gttext})';
    words(ismember(words, '-')) = [];
elseif strcmpi(opts.dataset, 'GW')
    wordsTe = data.words;
    words = unique({wordsTe.gttext})';
else
    error('Dataset not supported');
end

% Extracts the class of every word in the lexicon
% Class is equal to 0 if the word does not appear in the dataset
class_words = zeros(length(words),1);
[~,ia,ib] = inters(words,{wordsTe.gttext},'stable');
class_words(ia) = [wordsTe(ib).class];

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
lexicon.class_words = class_words;

save(opts.fileLexicon,'lexicon');

end

% Ugly hack to deal with the lack of stable intersection in old versions of
% matlab
function [empty1, ia, ib] = stableintersection(a, b, varargin)
empty1=0;
[~,ia,ib] = intersect(a,b);
[ia, tmp2] = sort(ia);
ib = ib(tmp2);
end