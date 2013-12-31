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

%% Learn common subspace, project and evaluate
[embedding,emb_repr] = learn_common_subspace(opts,data);