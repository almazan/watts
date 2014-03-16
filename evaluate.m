function results = evaluate(opts,data,embedding)

%% Load attribute representations
attReprTe = readMat(opts.fileAttRepresTe);

data.attReprTe = single(attReprTe);
data.phocsTe = single(data.phocsTe);
data.wordClsTe = single(data.wordClsTe);

% Augment phocs with length?
%W={data.wordsTe.gttext};
%data.phocsTe = [data.phocsTe;encodeWordsLength(W,10)];

% Evaluate the retrieval task
results.retrieval = evaluate_retrieval(opts,data,embedding);

% Evaluate the recognition task
if opts.evalRecog
    results.recognition = evaluate_recognition(opts,data,embedding);
end

end


