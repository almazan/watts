function opts = prepare_opts()

% Adding all the necessary libraries and paths
addpath('util/');
run('util/vlfeat-0.9.18/toolbox/vl_setup')

% Set random seed to default
rng('default');

% Select the dataset
opts.dataset = 'SVT';

opts.path_datasets = 'datasets';
opts.pathDataset = sprintf('%s/%s/',opts.path_datasets,opts.dataset);
opts.pathImages = sprintf('%s/%s/images/',opts.path_datasets,opts.dataset);
opts.pathDocuments = sprintf('%s/%s/documents/',opts.path_datasets,opts.dataset);
opts.pathQueries = sprintf('%s/%s/queries/',opts.path_datasets,opts.dataset);

% Options FV features
opts.numWordsTrainGMM = 500;
opts.SIFTDIM = 128;
opts.PCADIM = 62;
opts.numSpatialX = 6;
opts.numSpatialY = 2;
opts.G = 16;
opts.phowOpts = {'Verbose', false, 'Step', 3, 'FloatDescriptors', true, 'sizes',[2,4,6,8,10,12]} ;
opts.doMinibox = 1;
opts.minH = -1;
opts.fold = -1;

% Options PHOC attributes
opts.levels = [2 3 4 5];
opts.levelsB = [2];
opts.numBigrams = 50;
fid = fopen('data/bigrams.txt','r');
bgrams = textscan(fid,'%s');
fclose(fid);
opts.bgrams = bgrams{1}(1:opts.numBigrams);
opts.unigrams = 'abcdefghijklmnopqrstuvwxyz';
opts.digits='0123456789';
opts.considerDigits = 1;

% Options learning models
opts.bagging = 1;
opts.cluster = 0;
opts.sgdparams.eta0s = single([1]);
opts.sgdparams.lbds = single([1e-3,1e-4,1e-5]);
opts.sgdparams.betas = int32([32,64,80]);
opts.sgdparams.bias_multipliers = single([1]);
opts.sgdparams.epochs = 75;
opts.sgdparams.eval_freq = 2;
opts.sgdparams.t = 0;
opts.sgdparams.weightPos = 1;
opts.sgdparams.weightNeg = 1;

% Options embedding
opts.RemoveStopWords = 0;
opts.TestFV = 0;
opts.TestDirect = 1;

opts.TestPlatts = 0;
opts.Platts.verbose = 1;

opts.TestRegress = 0;
opts.Reg.Reg = [1e-1,1e-2,1e-3,1e-4];
opts.Reg.verbose = 1;

opts.TestCCA = 0;
opts.CCA.Dims = [96,128];
opts.CCA.Reg = [1e-4,1e-5,1e-6];
opts.CCA.verbose = 1;

opts.TestKCCA = 1;
opts.KCCA.M = [2500];
opts.KCCA.G = [40];
opts.KCCA.Dims = [192,256];
opts.KCCA.Reg = [1e-5,1e-6];
opts.KCCA.verbose = 1;

opts.evalRecog = 1;

% Specific dataset options
if strcmp(opts.dataset,'GW')
    opts.fold = 1;
    opts.evalRecog = 0;
elseif strcmp(opts.dataset,'IAM')
    opts.PCADIM = 30;
    opts.RemoveStopWords = 1;
    opts.swFile = 'data/swIAM.txt';
    opts.evalRecog = 0;
elseif strcmp(opts.dataset,'IIIT5K')
    opts.minH = 100;
    opts.doMinibox = 0;
elseif strcmp(opts.dataset,'SVT')
    opts.minH = 100;
    opts.doMinibox = 0;
elseif strcmp(opts.dataset,'ICDAR11')
    opts.minH = 100;
    opts.doMinibox = 0;
end

opts.FVdim = (opts.PCADIM+2)*opts.numSpatialX*opts.numSpatialY*opts.G*2;

if opts.evalRecog
    opts.TestKCCA = 1;
end

% Tags
tagminH = '';
if opts.minH > -1
    tagminH = sprintf('_minH%d',opts.minH);
end
tagFold = '';
if opts.fold > -1
    tagFold = sprintf('_fold%d',opts.fold);
end
tagPCA = sprintf('_PCA%d',opts.PCADIM);
tagGMM = sprintf('_GMM%dx%dx%dx%d%s',opts.G,opts.numSpatialY,opts.numSpatialX,opts.PCADIM+2,tagminH);
tagLevels = ['_l' strrep(strrep(strrep(mat2str(opts.levels),' ',''),'[',''),']','')];
tagLevelsB = ['_lb' strrep(strrep(strrep(mat2str(opts.levelsB),' ',''),'[',''),']','')];
tagNumB = sprintf('_nb%d',opts.numBigrams);
if opts.bagging
    tagBagging = '_bagging';
else
    tagBagging = '_noBagging';
end
tagFeats = '_FV';

opts.tagPHOC = sprintf('_PHOCs%s%s%s',tagLevels,tagLevelsB,tagNumB);
opts.tagFeatures = sprintf('%s%s%s%s',tagFeats,tagPCA,tagGMM,tagFold);

% Paths and files
opts.pathData = './data/files';
if ~exist(opts.pathData,'dir')
    mkdir(opts.pathData);
end
opts.dataFolder = sprintf('%s/%s%s%s',opts.pathData,opts.dataset,opts.tagPHOC,opts.tagFeatures);
if ~exist(opts.dataFolder,'dir')
    mkdir(opts.dataFolder);
end
opts.fileData = sprintf('%s/%s_data.mat',opts.pathData,opts.dataset);
opts.fileImages = sprintf('%s/%s_images.mat',opts.pathData,opts.dataset);
opts.fileWriters = sprintf('%s/%s_writers.mat',opts.pathData,opts.dataset);
opts.fileGMM = sprintf('%s/%s%s.mat',opts.dataFolder,opts.dataset,tagGMM);
opts.filePCA = sprintf('%s/%s%s.mat',opts.dataFolder,opts.dataset,tagPCA);
opts.filePHOCs = sprintf('%s/%s%s.mat',opts.dataFolder,opts.dataset,opts.tagPHOC);
opts.fileFeatures = sprintf('%s/%s%s.mat',opts.dataFolder,opts.dataset,opts.tagFeatures);
opts.fileAttModels = sprintf('%s/%s_attModels%s%s%s.mat',opts.dataFolder,opts.dataset,opts.tagPHOC,opts.tagFeatures,tagBagging);
opts.fileAttRepres = sprintf('%s/%s_attRepres%s%s%s.mat',opts.dataFolder,opts.dataset,opts.tagPHOC,opts.tagFeatures,tagBagging);
opts.folderModels = sprintf('%s/models%s/',opts.dataFolder,tagBagging);
opts.modelsLog = sprintf('%s/learning.log',opts.folderModels);
if ~exist(opts.folderModels,'dir')
    mkdir(opts.folderModels);
end
opts.fileSets = sprintf('data/%s_words_indexes_sets%s.mat',opts.dataset,tagFold);
opts.fileLexicon = sprintf('%s/%s_lexicon%s.mat',opts.pathData,opts.dataset,opts.tagPHOC);
end