function feats = extract_FV_features_fast(opts)

step =  opts.phowOpts{find(strcmp(opts.phowOpts,'Step'))+1};
sizes = int32(opts.phowOpts{find(strcmp(opts.phowOpts,'sizes'))+1});
computeFV_mex(opts.fileImages, opts.filePCA, opts.fileGMM, step,sizes, opts.doMinibox,opts.fileFeatures );
