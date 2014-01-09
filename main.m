%% Word Spotting and Recognition with Embedded Attributes
% Authors: Jon Almazan and Albert Gordo
% Contact: almazan@cvc.uab.es

%% Prepare options and read dataset
opts = prepare_opts();
data = load_dataset(opts);

%% Embed text labels with PHOC
data.phocs = embed_labels_PHOC(opts,data);

%% Extract features from images
extract_features(opts);

%% Split data into sets
data = prepare_data_learning(opts,data);

%% Learn PHOC attributes
data.att_models = learn_attributes(opts,data);

%% Learn common subspaces and/or calibrations
[embedding,mAPsval] = learn_common_subspace(opts,data);

%% Evaluate
mAPstest = evaluate(opts,data,embedding);