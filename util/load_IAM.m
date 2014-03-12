function data = load_IAM(opts)
disp('* Reading IAM info *');

%% Reading writers information
file=[opts.pathDataset 'forms.gtp'];
fid = fopen(file, 'r');

input = textscan(fid, '%s %d %d %s %d %d %d %d');
nDocs = length(input{1});
numWriters = max(input{2}(:))+1;

docs = containers.Map();
list = cell(numWriters,1);
for i=1:nDocs
    % Determine the class of the query given the GT text
    idW = input{2}(i) + 1;
    docs(input{1}{i}) = idW;
    list{idW}{length(list{idW})+1} = input{1}{i};
end
writers.list = list;
writers.docs = docs;
writers.num = length(list);
fclose(fid);
data.writers = writers;

%% Reading words information
fileQueries=[opts.pathQueries 'queries.gtp'];
fid = fopen(fileQueries, 'r');
input = textscan(fid, '%s %d %d %d %d %s %s');
nWords = length(input{1});

margin=16;
pathIm = '';

for j=1:nWords
    words(j).pathIm = [opts.pathImages input{1}{j}];
    words(j).pathIm = [words(j).pathIm '.png'];
    loc = [input{2}(j) input{4}(j) input{3}(j) input{5}(j)];
    locOrig = loc;
    words(j).origLoc = loc;
    
    loc = [loc(1)-margin loc(2)+margin loc(3)-margin loc(4)+margin];
    if ~strcmp(words(j).pathIm,pathIm)
        imDoc = imread(words(j).pathIm);
        if ndims(imDoc)>2
            imDoc = rgb2gray(imDoc);
        end
        pathIm = words(j).pathIm;
    end
    [H,W] = size(imDoc);
    x1 = max(loc(1),1); x2 = min(loc(2),W);
    y1 = max(loc(3),1); y2 = min(loc(4),H);
    im = imDoc(y1:y2,x1:x2);
    words(j).loc = [x1 x2 y1 y2];
    [words(j).H,words(j).W] = size(im);
    words(j).gttext = input{6}{j};
    
    words(j).docId = input{1}{j};
    wordId = input{7}{j};
    words(j).wordId = wordId;
    words(j).lineId = wordId(1:end-3);    
end

newClass = 1;
words(1).class = [];
classes = containers.Map();
idxClasses = {};
names = {};

for i=1:length(words)
    gttext = lower(words(i).gttext);
    % Determine the class of the query given the GT text
    if isKey(classes, gttext)
        class = classes(gttext);
    else
        class = newClass;
        newClass = newClass+1;
        classes(gttext) = class;
        idxClasses{class} = int32([]);
        names{class} = gttext;
    end
    idxClasses{class} = [idxClasses{class} i];
    words(i).class = class;
end

%% Update writer information
for i=1:length(words)
    docId = words(i).docId;
    words(i).writerId = writers.docs(docId);
end

%% Output
data.words = words;
data.classes = classes;
data.idxClasses = idxClasses;
data.names = names;

end