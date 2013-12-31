function [att_models,attFeatsTr] = learn_attributes_bagging(opts,data)
% Learns models using several folds of the training data.
% Also produces the scores of the training samples with an unbiased model.
% For each training sample, we get its score using a model constructed as
% the average of all models that did not use that particular training
% sample for training. This is done online.

% Set params. eta, lbd, beta and bias_multiplier can receive multiple
% values that will be crossvalidated for map.
params = opts.sgdparams;

feats = data.feats_training;
phocs = data.phocs_training;

[numAtt,numSamples] = size(phocs);
dimFeats = size(feats,1);

% Output encoded
attFeatsTr = single(zeros(numAtt,numSamples));
att_models = struct('W',[],'B',[],'numPosSamples',[]);

% For each attribute
for idxAtt = 1:numAtt
    fileModel = sprintf('%smodel_%.3d.mat',opts.folderModels,idxAtt);
    if ~exist(fileModel,'file')
        % Prepare the output classifier and bias
        W=single(zeros(dimFeats,1));
        B = 0;
        % Keep counts of how many updates, global and per sample
        Np = zeros(numSamples,1);
        it = 0;
        N = 0;
        numPosSamples = 0;
        
        idxPos = find(phocs(idxAtt,:)~=0);
        idxNeg = find(phocs(idxAtt,:)==0);
        
        % We want to do at least 10 iterations. However, some iterations do not
        % do anything, so we aim at 20 iterations.
        if length(idxPos)>=2
            while (N<=10 && it<=20)
                it = it + 1;
                fprintf('\nStarting with attribute %d iteration %d\n',idxAtt,it);
                
                pp = randperm(length(idxPos));
                pn = randperm(length(idxNeg));
                idxTrain =[ idxPos(pp(1:floor(length(pp)*0.8)))  idxNeg(pn(1:floor(length(pn)*0.8)))];
                idxVal = [idxPos(pp(1+floor(length(pp)*0.8):end) )  idxNeg(pn(1+floor(length(pn)*0.8):end))];
                idxTrain = idxTrain(randperm(length(idxTrain)));
                
                featsTrain = feats(:,idxTrain);
                phocsTrain = phocs(:,idxTrain);
                featsVal = feats(:,idxVal);
                phocsVal = phocs(:,idxVal);
                
                labelsTrain = int32(phocsTrain(idxAtt,:)~=0);
                labelsVal = int32(phocsVal(idxAtt,:)~=0);
                nPosT = sum(labelsTrain);
                nPosV = sum(labelsVal);
                
                numPosSamples = numPosSamples + nPosT;
                
                if nPosT==0
                    fprintf('No positive training samples for att %d it %d. Skipping\n',idxAtt,it);
                    continue;
                end
                if nPosV==0
                    fprintf('No positive validation samples for att %d it %d. Skipping\n',idxAtt,it);
                    continue;
                end
                
                % Learn model
                tic;
                modelAtt = sgdsvm_train_cv_mex(featsTrain,labelsTrain,featsVal,labelsVal,params);
                t=toc;
                fprintf('Model for attribute %d it %d (%.2f map) learned in %.0f seconds using %d positive samples\n',idxAtt, it, modelAtt.info.acc, t, nPosT);
                f=fopen('learning.log','a');
                fprintf(f,'Model for attribute %d it %d (%.2f map) learned in %.0f seconds using %d positive samples\n',idxAtt,it, modelAtt.info.acc, t, nPosT);
                fclose(f);
                
                % Update things. Update the scores of the samples not used for
                % training, as well as the global model.
                N = N+1;
                Np(idxVal) = Np(idxVal)+1;
                sc = modelAtt.W'*featsVal;
                attFeatsTr(idxAtt,idxVal) = attFeatsTr(idxAtt,idxVal) + sc;
                W = W + modelAtt.W;
                B = B + modelAtt.B;
                
            end
            
            
            % Check that all the samples has been used at least once for validation
            % If there is some, we will learn a new model using them
            
            while ~isempty(find(Np<=1))
                fprintf('%d samples that only once or less for validation\n',sum(Np<=1));
                idxVal = find(Np<=1)';
                % idxVal may not contain any positive. If it does not, add 20%
                % of positives
                if isempty(find(phocs(idxAtt,idxVal)~=0))
                    idxPos = find(phocs(idxAtt,:)~=0);
                    pp = randperm(length(idxPos));
                    idxVal = [idxVal idxPos(pp(1+floor(length(pp)*0.8):end) ) ];
                end
                % It is not possible that idxVal contains all the positives, since 20 percent are picked at every iteration, so
                % in 10 iterations there must be at least one positive sample
                % that has been picked twice (pigeonhole principle!)
                
                idxTrain = setdiff(1:numSamples,idxVal);
                idxTrain = idxTrain(randperm(length(idxTrain)));
                
                featsTrain = feats(:,idxTrain);
                phocsTrain = phocs(:,idxTrain);
                featsVal = feats(:,idxVal);
                phocsVal = phocs(:,idxVal);
                
                labelsTrain = int32(phocsTrain(idxAtt,:)~=0);
                labelsVal = int32(phocsVal(idxAtt,:)~=0);
                nPosT = sum(labelsTrain);
                nPosV = sum(labelsVal);
                
                numPosSamples = numPosSamples + nPosT;
                
                if nPosT==0
                    fprintf('No positive training samples for att %d it %d. Skipping\n',idxAtt,it);
                    continue;
                end
                if nPosV==0
                    fprintf('No positive validation samples for att %d it %d. Skipping\n',idxAtt,it);
                    continue;
                end
                N = N + 1;
                
                
                % Learn model
                tic;
                modelAtt = sgdsvm_train_cv_mex(featsTrain,labelsTrain,featsVal,labelsVal,params);
                t=toc;
                fprintf('Model for attribute %d it %d (%.2f map) learned in %.0f seconds using %d positive samples\n',idxAtt, it, modelAtt.info.acc, t, nPosT);
                f=fopen('learning.log','a');
                fprintf(f,'Model for attribute %d it %d (%.2f map) learned in %.0f seconds using %d positive samples\n',idxAtt,it, modelAtt.info.acc, t, nPosT);
                fclose(f);
                
                % Update things. Update the scores of the samples not used for
                % training, as well as the global model.
                Np(idxVal) = Np(idxVal)+1;
                sc = modelAtt.W'*featsVal;
                attFeatsTr(idxAtt,idxVal) = attFeatsTr(idxAtt,idxVal) + sc;
                W = W + modelAtt.W;
                B = B + modelAtt.B;
                
            end
        end
        
        % Average and save
        model.W = W;
        model.B = B;
        model.numPosSamples = 0;
        if N~=0
            model.W = model.W/N;
            model.B = model.B/N;
            attFeatsTr(idxAtt,:) = attFeatsTr(idxAtt,:) ./ Np';
            model.numPosSamples = ceil(numPosSamples / N);
        end
        
        attFeatsBag = attFeatsTr(idxAtt,:);
        save(fileModel,'model','attFeatsBag');
        
        att_models(idxAtt) = model;
    else
        fprintf('\nAttribute %d already computed. Loaded.\n',idxAtt);
        load(fileModel);
        att_models(idxAtt) = model;
        attFeatsTr(idxAtt,:) = attFeatsBag;
    end
end

end