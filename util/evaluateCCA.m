function [embedding,emb_repr,mAP] = evaluateCCA(opts,DATA)
% Evaluate CCA.

%% Part 1: Crosvalidate to find the best parameters in the config range
fprintf('\n');
disp('**************************************');
disp('*************   CV CCA   *************');
disp('**************************************');

% A) L2 normalize and mean center. Not critical, but helps a bit.
attReprTr = bsxfun(@rdivide, DATA.attReprTr,sqrt(sum(DATA.attReprTr.*DATA.attReprTr)));
attReprTr(isnan(attReprTr)) = 0;
attReprTrFull = bsxfun(@rdivide, DATA.attReprTrFull,sqrt(sum(DATA.attReprTrFull.*DATA.attReprTrFull)));
attReprTrFull(isnan(attReprTrFull)) = 0;
attReprVal = bsxfun(@rdivide, DATA.attReprVa,sqrt(sum(DATA.attReprVa.*DATA.attReprVa)));
attReprVal(isnan(attReprVal)) = 0;
attReprTe = bsxfun(@rdivide, DATA.attReprTe,sqrt(sum(DATA.attReprTe.*DATA.attReprTe)));
attReprTe(isnan(attReprTe)) = 0;

phocsTrFull = bsxfun(@rdivide, DATA.phocsTrFull,sqrt(sum(DATA.phocsTrFull.*DATA.phocsTrFull)));
phocsTr = bsxfun(@rdivide, DATA.phocsTr,sqrt(sum(DATA.phocsTr.*DATA.phocsTr)));
phocsVal = bsxfun(@rdivide, DATA.phocsVa,sqrt(sum(DATA.phocsVa.*DATA.phocsVa)));
phocsTe = bsxfun(@rdivide, DATA.phocsTe,sqrt(sum(DATA.phocsTe.*DATA.phocsTe)));

matts = mean(attReprTr,2);
attReprTrFull = bsxfun(@minus,attReprTrFull, matts);
attReprTr = bsxfun(@minus,attReprTr, matts);
attReprVal = bsxfun(@minus,attReprVal, matts);
attReprTe =  bsxfun(@minus, attReprTe,matts);

mphocs = mean(phocsTr,2);
phocsTrFull = bsxfun(@minus, phocsTrFull,mphocs);
phocsTr = bsxfun(@minus, phocsTr,mphocs);
phocsVal = bsxfun(@minus, phocsVal,mphocs);
phocsTe=  bsxfun(@minus, phocsTe,mphocs);

% B) For each config, learn on Tr (small), validate on val, and keep the
% best according to QBE+QBS map. Other criterions (eg, QBE map or QBS p1) are
% possible.

bestScore = -1;
bestReg = opts.CCA.Reg(1);
bestK = opts.CCA.Dims(1);

for reg=opts.CCA.Reg
    % Learn CCA using tr (small)
    [Wx,Wy,r] = cca(attReprTr', phocsTr',reg);
    for K=opts.CCA.Dims
        % Check that there are enough valid projections
        if  ~isreal(Wx(:,1:K)) || ~isreal(Wy(:,1:K))
            continue;
        end
        
        % Embed val and test
        attReprVal_cca = Wx(:,1:K)' * attReprVal;
        phocsVal_cca = Wy(:,1:K)' * phocsVal;
        attReprTe_cca = Wx(:,1:K)' * attReprTe;
        phocsTe_cca = Wy(:,1:K)' * phocsTe;
        
        % L2 normalize (critical)
        attReprVal_cca = (bsxfun(@rdivide, attReprVal_cca, sqrt(sum(attReprVal_cca.*attReprVal_cca))));
        attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
        phocsVal_cca = (bsxfun(@rdivide, phocsVal_cca, sqrt(sum(phocsVal_cca.*phocsVal_cca))));
        phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));
        
        % Get QBE and QBS scores on validation and test. Note that looking
        % at the test ones is frowned upon. Just for debugging purposes.
        % Evaluate
        % Val
        % QBE
        [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprVal_cca, attReprVal_cca,DATA.wordClsVa,DATA.labelsVa);
        qbe_val_map = mean(mAPEucl);
        qbe_val_p1 = mean(p1);
        
        % QBS (note the 1 at the end)
        [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsVal_cca, attReprVal_cca,DATA.wordClsVa,DATA.labelsVa,1);
        qbs_val_map = mean(mAPEucl);
        qbs_val_p1 = mean(p1);
        
        % Test
        % QBE
        [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
        qbe_test_map = mean(mAPEucl);
        qbe_test_p1 = mean(p1);
        
        % QBS (note the 1 at the end)
        [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
        qbs_test_map = mean(mAPEucl);
        qbs_test_p1 = mean(p1);
        
        if opts.CCA.verbose
            % Display info
            disp('----------Test results only for debug purposes. Do not use!--------------');
            fprintf('reg: %.8f. k: %d\n',  reg, K);
            fprintf('qbe -- val: (map: %.2f. p@1: %.2f).  test: (map: %.2f. p@1: %.2f)\n', 100*qbe_val_map, 100*qbe_val_p1, 100*qbe_test_map, 100*qbe_test_p1);
            fprintf('qbs -- val: (map: %.2f. p@1: %.2f).  test: (map: %.2f. p@1: %.2f)\n', 100*qbs_val_map, 100*qbs_val_p1, 100*qbs_test_map, 100*qbs_test_p1);
            disp('------------------------------------');
        end
        
        % Better than before? update.
        if qbe_val_map + qbs_val_map > bestScore
            bestScore = qbe_val_map + qbs_val_map;
            bestK = K;
            bestReg = reg;
        end
    end
end
disp('------------------------------------');
fprintf('Best qbe map result on validation: %.2f map with reg %.8f and K %d\n', 100*bestScore, bestReg, bestK);
disp('------------------------------------');

%% Part 2: Use the best parameters on the whole thing
fprintf('\n');
disp('**************************************');
disp('***********   CCA FINAL  *************');
disp('**************************************');
% Learn CCA
[Wx,Wy,r] = cca(attReprTrFull', phocsTrFull',bestReg);
K = bestK;
% Embed
% Embed  test
attReprTe_cca = Wx(:,1:K)' * attReprTe;
phocsTe_cca = Wy(:,1:K)' * phocsTe;

% L2 normalize (critical)
attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
qbs_test_map = mean(mAPEucl);
qbs_test_p1 = mean(p1);

% Display info
fprintf('\n');
disp('------------------------------------');
fprintf('reg: %.8f. k: %d\n',  bestReg, bestK);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

mAP = (qbe_test_map+qbs_test_map)/2;
emb_repr.attRepr_emb = attReprTe_cca;
emb_repr.phocRepr_emb = phocsTe_cca;
embedding.Wx = Wx;
embedding.Wy = Wy;
embedding.K = K;
embedding.reg = bestReg;
embedding.matts = matts;
embedding.mphocs = mphocs;

end
