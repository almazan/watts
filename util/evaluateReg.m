function  mAP = evaluateReg(opts,DATA,embedding)
% Evaluate Reg.

fprintf('\n');
disp('**************************************');
disp('***********  Regression   ************');
disp('**************************************');

% A) L2 normalize and mean center. Not critical, but helps a bit.
attReprTe = bsxfun(@rdivide, DATA.attReprTe,sqrt(sum(DATA.attReprTe.*DATA.attReprTe)));
attReprTe(isnan(attReprTe)) = 0;
phocsTe = bsxfun(@rdivide, DATA.phocsTe,sqrt(sum(DATA.phocsTe.*DATA.phocsTe)));

attReprTe =  bsxfun(@minus, attReprTe, embedding.matts);
phocsTe=  bsxfun(@minus, phocsTe, embedding.mphocs);

% Embed  test
attReprTe_reg = embedding.Wx(:,1:embedding.K)' * attReprTe;
phocsTe_reg =  phocsTe;

% L2 normalize (critical)
attReprTe_reg = (bsxfun(@rdivide, attReprTe_reg, sqrt(sum(attReprTe_reg.*attReprTe_reg))));
phocsTe_reg = (bsxfun(@rdivide, phocsTe_reg, sqrt(sum(phocsTe_reg.*phocsTe_reg))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_reg,attReprTe_reg,DATA.wordClsTe,DATA.labelsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_reg,attReprTe_reg,DATA.wordClsTe,DATA.labelsTe,1);
qbs_test_map = mean(mAPEucl);
qbs_test_p1 = mean(p1);

% Display info
disp('------------------------------------');
fprintf('reg: %.8f. k: %d\n',  embedding.reg, embedding.K);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

mAP.qbe = 100*qbe_test_map;
mAP.qbs = 100*qbs_test_map;
end