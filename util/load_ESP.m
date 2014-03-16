function data = load_ESP(opts)
disp('* Reading ESP info *');


%% Reading words information
fileQueries=[opts.pathQueries 'queries.gtp'];
fid = fopen(fileQueries, 'r');

input = textscan(fid, '%s %d %d %d %d %s');
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
    [words(j).H,words(j).W,numC] = size(im);
    words(j).gttext = input{6}{j};
    
    words(j).docId = input{1}{j};
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

%% Output
data.words = words;
data.classes = classes;
data.idxClasses = idxClasses;
data.names = names;

end
