function mAP = evaluatePlatts(opts, DATA, platts)
%% Platts
fprintf('\n');
disp('**************************************');
disp('*************   PLATTS   *************');
disp('**************************************');

% Apply calibration
attReprTePlatts = 1./(1 + exp(bsxfun(@plus, bsxfun(@times, DATA.attReprTe, platts(:,1)), platts(:,2))));

% L2 normalize. Extremely important
phocsTeL2 = bsxfun(@rdivide, DATA.phocsTe,sqrt(sum(DATA.phocsTe.*DATA.phocsTe)));
attReprTeL2 = bsxfun(@rdivide, attReprTePlatts,sqrt(sum(attReprTePlatts.*attReprTePlatts)));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTeL2,attReprTeL2,DATA.wordClsTe,DATA.labelsTe);
qbe_map = mean(mAPEucl);
qbe_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTeL2,attReprTeL2,DATA.wordClsTe,DATA.labelsTe,1);
qbs_map = mean(mAPEucl);
qbs_p1 = mean(p1);

disp('------------------------------------');
fprintf('Dimensions: %d\n',size(attReprTeL2,1));
fprintf('qbe -- test: (map: %.2f. p@1: %.2f)\n',  100*qbe_map, 100*qbe_p1);
fprintf('qbs -- test: (map: %.2f. p@1: %.2f)\n',  100*qbs_map, 100*qbs_p1);
disp('------------------------------------');

mAP.qbe = 100*qbe_map;
mAP.qbs = 100*qbs_map;
end

