function [embedding,mAP,bestmAPval] = learn_common_subspace(opts,data)

%% Load attribute representations
attReprTr = readMat(opts.fileAttRepresTr);
attReprVa = readMat(opts.fileAttRepresVal);


data.attReprTr = single(attReprTr);
data.phocsTr = single(data.phocsTr);
data.wordClsTr = single(data.wordClsTr);

keep = sum(data.phocsTr)~=0;
data.attReprTr = data.attReprTr(:,keep);
data.phocsTr = data.phocsTr(:,keep);
data.wordClsTr = data.wordClsTr(keep);
data.labelsTr = data.labelsTr(keep);

% Augment phocs with length?
%W={data.wordsTr(keep).gttext};
%data.phocsTr = [data.phocsTr;encodeWordsLength(W,10)];

%% Randomize train
p = randperm(size(data.attReprTr,2));
data.attReprTr = data.attReprTr(:,p);
data.phocsTr = data.phocsTr(:,p);
data.wordClsTr = data.wordClsTr(p);
data.labelsTr = data.labelsTr(p);
data.TestPermutation = p;

%% Create "fake" validation partition
% Extract 30% descs from train to use as validation.
% We also keep the original full training set to retrain later on.

data.attReprTrFull = data.attReprTr;
data.phocsTrFull = data.phocsTr;
data.wordClsTrFull = data.wordClsTr;
data.labelsTrFull = data.labelsTr;

nval = floor(0.3*size(data.attReprTr,2));
data.attReprVa = data.attReprTr(:,1:nval);
data.attReprTr = data.attReprTr(:,nval+1:end);
data.phocsVa = data.phocsTr(:,1:nval);
data.phocsTr = data.phocsTr(:,nval+1:end);
data.wordClsVa = data.wordClsTr(1:nval);
data.wordClsTr = data.wordClsTr(nval+1:end);
data.labelsVa = data.labelsTr(1:nval);
data.labelsTr = data.labelsTr(nval+1:end);

mAP.reg = 0;
mAP.cca = 0;
mAP.kcca = 0;
embedding.platts = [];
embedding.reg = [];
embedding.cca = [];
embedding.kcca = [];
bestmAPval = 0;

if opts.TestPlatts
    embedding.platts = learnPlatts(opts, data);
end

if opts.TestRegress
    [embedding.reg,mAP.reg]= learnReg(opts, data);
    if mAP.reg > bestmAPval
        bestmAPval = mAP.reg;
    end
end

if opts.TestCCA
    [embedding.cca,mAP.cca] = learnCCA(opts, data);
    if mAP.cca > bestmAPval
        bestmAPval = mAP.cca;
    end
end

if opts.TestKCCA
    [embedding.kcca,mAP.kcca] = learnKCCA(opts, data);
    if mAP.kcca > bestmAPval
        bestmAPval = mAP.kcca;
    end
end

end