function  mAP = evaluateKCCA(opts,DATA,embedding)
% Evaluate KCCA.

fprintf('\n');
disp('**************************************');
disp('***************  KCSR   **************');
disp('**************************************');

matx = embedding.rndmatx(1:embedding.M,:);
maty = embedding.rndmaty(1:embedding.M,:);

tmp = matx*DATA.attReprTe;
attReprTe_emb = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
tmp = maty*DATA.phocsTe;
phocsTe_emb = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];

% Mean center
attReprTe_emb=bsxfun(@minus, attReprTe_emb, embedding.matts);
phocsTe_emb=bsxfun(@minus, phocsTe_emb, embedding.mphocs);

% Embed test
attReprTe_cca = embedding.Wx(:,1:embedding.K)' * attReprTe_emb;
phocsTe_cca = embedding.Wy(:,1:embedding.K)' * phocsTe_emb;

% L2 normalize (critical)
attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
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