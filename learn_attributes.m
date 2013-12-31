function attModels = learn_attributes(opts,data)
% Learns the attribute models and projects the dataset to the new attribute
% space.
if ~exist(opts.fileAttModels,'file')
    
    if opts.bagging
        if opts.cluster
        else
            load(opts.fileFeatures,'features');
            data.feats_training = [features(:,data.idxTrain) features(:,data.idxValidation)];
            data.phocs_training = [data.phocsTr data.phocsVa];
            [attModels,attReprTr] = learn_attributes_bagging(opts,data);
        end
    end
    
    W = [attModels(:).W];
    attReprVa= [];
    if ~exist('features','var');
        load(opts.fileFeatures,'features');
    end
    feats_te = features(:,data.idxTest);
    attReprTe = W'*feats_te;
    
    save(opts.fileAttRepres,'attReprTr','attReprVa','attReprTe');
    save(opts.fileAttModels,'attModels');
else
    load(opts.fileAttModels);
end
end