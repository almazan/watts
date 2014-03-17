function [ wer ] = compute_wer( matfile )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

load(matfile)

% Discard ungoody words
linesTe = linesTe(idx);
p1 = p1(idx);

% Create dictionary
dict =java.util.Hashtable;

for i=1:length(linesTe)
    dict.put(linesTe{i}, [dict.get(linesTe{i}); i]);
end

entries = dict.entrySet.toArray;

wer = zeros(1, length(entries));

for i=1:length(entries)
    idxL = entries(i).getValue;
    wer(i) = mean(p1(idxL));
end
wer = 1-mean(wer);
end

