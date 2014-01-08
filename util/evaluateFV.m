function  mAP = evaluateFV(opts,DATA)
%% Direct FV
fprintf('\n');
disp('**************************************');
disp('*************   FV   *************');
disp('**************************************');

load(opts.fileFeatures,'features');                       
feats_te = features(:,DATA.idxTest);

%Evaluate
% QBE
[p1,mAPEucl,q] = eval_dp_asymm(opts,feats_te,feats_te,DATA.wordClsTe,DATA.labelsTe);
qbe_map = mean(mAPEucl);
qbe_p1 = mean(p1);

fprintf('Dimensions: %d\n',size(feats_te,1));
fprintf('qbe -- test: (map: %.2f. p@1: %.2f)\n',  100*qbe_map, 100*qbe_p1);

mAP = qbe_map;
end