function  mAP = evaluateDirect(opts,DATA)
%% Direct approach (No Platts/CCA/KCCA)
fprintf('\n');
disp('**************************************');
disp('*************   DIRECT   *************');
disp('**************************************');

% For the direct approach, we always L2 normalize
phocsTeL2 = bsxfun(@rdivide, DATA.phocsTe,sqrt(sum(DATA.phocsTe.*DATA.phocsTe)));
attFeatsTeL2 = bsxfun(@rdivide,DATA.attReprTe,sqrt(sum(DATA.attReprTe.*DATA.attReprTe)));

%Evaluate
% QBE
[p1,mAPEucl,q] = eval_dp_asymm(opts,attFeatsTeL2,attFeatsTeL2,DATA.wordClsTe,DATA.labelsTe);
qbe_map = mean(mAPEucl);
qbe_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl,q] = eval_dp_asymm(opts,phocsTeL2,attFeatsTeL2,DATA.wordClsTe,DATA.labelsTe,1);
qbs_map = mean(mAPEucl);
qbs_p1 = mean(p1);

disp('------------------------------------');
fprintf('Dimensions: %d\n',size(attFeatsTeL2,1));
fprintf('qbe -- test: (map: %.2f. p@1: %.2f)\n',  100*qbe_map, 100*qbe_p1);
fprintf('qbs -- test: (map: %.2f. p@1: %.2f)\n',  100*qbs_map, 100*qbs_p1);
disp('------------------------------------');

mAP.qbe = 100*qbe_map;
mAP.qbs = 100*qbs_map;
end

