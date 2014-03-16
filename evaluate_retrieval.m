function mAP = evaluate_retrieval(opts,data,embedding)

mAP.fv = [];
mAP.direct = [];
mAP.platts = [];
mAP.reg = [];
mAP.cca = [];
mAP.kcca = [];

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

% % Hybrid spotting not implemented yet
% if opts.evalHybrid
%     alpha = 0:0.1:1;
%     hybrid_test_map = zeros(length(alpha),1);
%     for i=1:length(alpha)
%         attRepr_hybrid = attReprTe_cca*alpha(i) + phocsTe_cca*(1-alpha(i));
%         [p1,mAPEucl,q] = eval_dp_asymm(opts,attRepr_hybrid, attReprTe_cca,DATA.queriesClassesTe,DATA.wordsTe);
%         hybrid_test_map(i) = mean(mAPEucl);
%     end
% end

end