% Load data
% Important. Positive class should have label 1. Negative classes may have
% any other integer label different from 1.
% Data must be in single format, labels in int32 format.
Xtrain = single(load('../datasets/ijcnn1/ijcnn1.tr.data'));
Ltrain = int32(load('../datasets/ijcnn1/ijcnn1.tr.labels'));
Xvalid = single(load('../datasets/ijcnn1/ijcnn1.val.data'));
Lvalid = int32(load('../datasets/ijcnn1/ijcnn1.val.labels'));
Xtest = single(load('../datasets/ijcnn1/ijcnn1.t.data'));
Ltest = int32(load('../datasets/ijcnn1/ijcnn1.t.labels'));
% 
% % Column vectors
Xtrain = Xtrain';
Xvalid = Xvalid';
Xtest = Xtest';

% Set params. eta, lbd, beta and bias_multiplier can receive multiple
% values that will be crossvalidated form map.
params.eta0s = single([2,1,1e-1,1e-2]);
params.lbds = single([ 1e-2,1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8]);
params.betas = int32([32,64,128,256,512]);
params.bias_multipliers = single([1]);
params.epochs = 100;
params.eval_freq = 2;
params.t = 0;
params.weightPos = 1;
params.weightNeg = 0.5;

% Learn model
tic;
model = sgdsvm_train_cv_mex(Xtrain,Ltrain,Xvalid,Lvalid,params);
toc
model.info


% Evaluate model in validation and test sets
s = model.W'*Xvalid + model.B;
prob = 1./(1 + exp(s*model.PlattsA + model.PlattsB));
% recall of positives
r1 =sum(Lvalid(s>=0)==1)/sum(Lvalid==1);
% precision of positives
p1 =sum(Lvalid(s>=0)==1)/length(find(s>=0));
% recall of negatives
r0 =sum(Lvalid(s<0)==-1)/sum(Lvalid==-1);
% precision of negatives
p0 =sum(Lvalid(s<0)==-1)/length(find(s<0));
% map:
[a,b] = sort(s,'descend');
l = Lvalid(b);
map = sum((l==1).*(cumsum(l==1)./(1:length(l))'))/length(find(l==1));
fprintf(1, 'Results on validation set\n');
fprintf(1, 'p1: %.2f r1: %.2f p-1: %.2f r-1: %.2f map: %.2f\n',100*p1,100*r1,100*p0,100*r0,100*map);

s = model.W'*Xtest + model.B;
prob = 1./(1 + exp(s*model.PlattsA + model.PlattsB));
r1 =sum(Ltest(s>=0)==1)/sum(Ltest==1);
p1 =sum(Ltest(s>=0)==1)/length(find(s>=0));
r0 =sum(Ltest(s<0)==-1)/sum(Ltest==-1);
p0 =sum(Ltest(s<0)==-1)/length(find(s<0));
[a,b] = sort(s,'descend');
l = Ltest(b);
map = sum((l==1).*(cumsum(l==1)./(1:length(l))'))/length(find(l==1));
fprintf(1, 'Results on test set\n');
fprintf(1, 'p1: %.2f r1: %.2f p-1: %.2f r-1: %.2f map: %.2f\n',100*p1,100*r1,100*p0,100*r0,100*map);
