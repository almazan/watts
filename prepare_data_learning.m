function data = prepare_data_learning(opts,data)
load(opts.fileSets);

data.idxTrain = idxTrain;
data.idxValidation = idxValidation;
data.idxTest = idxTest;

data.wordsTr = data.words(idxTrain);
data.numWTr = length(data.wordsTr);
data.wordsVa = data.words(idxValidation);
data.numWVa = length(data.wordsVa);
data.wordsTe = data.words(idxTest);
data.numWTe = length(data.wordsTe);

data.labelsTr = {data.wordsTr(:).gttext};
data.labelsVa = {data.wordsVa(:).gttext};
data.labelsTe = {data.wordsTe(:).gttext};

data.wordClsTr = [data.wordsTr(:).class];
data.wordClsVa = [data.wordsVa(:).class];
data.wordClsTe = [data.wordsTe(:).class];

data.phocsTr = data.phocs(:,idxTrain);
data.phocsVa = data.phocs(:,idxValidation);
data.phocsTe = data.phocs(:,idxTest);
end