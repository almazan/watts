function [ encoded ] = encodeWordsLength( words, n )
lf = @(x) [ones(min(n,length(x)),1);zeros(n-min(n,length(x)),1)];
encoded = cellfun(lf, words, 'uniformoutput', false); 
encoded = single(cell2mat(encoded));
end

