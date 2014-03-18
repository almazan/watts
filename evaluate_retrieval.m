function mAP = evaluate_retrieval(opts,data,embedding)

mAP.fv = [];
mAP.direct = [];
mAP.platts = [];
mAP.reg = [];
mAP.cca = [];
mAP.kcca = [];
mAP.hybrid = [];

if opts.TestFV
    mAP.fv = evaluateFV(opts, data);
end

if opts.TestDirect
    mAP.direct = evaluateDirect(opts, data);
end

if opts.TestPlatts
    mAP.platts = evaluatePlatts(opts, data, embedding.platts);
end

if opts.TestRegress
    mAP.reg = evaluateReg(opts, data, embedding.reg);
end

if opts.TestCCA
    mAP.cca = evaluateCCA(opts, data, embedding.cca);
end

if opts.TestKCCA
    mAP.kcca = evaluateKCCA(opts, data, embedding.kcca);
end

if opts.TestHybrid
    mAP.hybrid = evaluateHybrid(opts, data, embedding.kcca);
end

end