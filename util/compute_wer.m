function [ wer ] = compute_wer(linesTe ,p1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


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
wer = 100*(1-mean(wer));
end

