function  [embedding,emb_repr,mAP] = evaluateReg(opts,DATA)
% Evaluate Reg.

%% Part 1: Crosvalidate to find the best parameters in the config range
fprintf('\n');
disp('**************************************');
disp('*************   CV Reg   *************');
disp('**************************************');

% A) L2 normalize and mean center. Not critical, but helps a bit.
attReprTr = bsxfun(@rdivide, DATA.attReprTr,sqrt(sum(DATA.attReprTr.*DATA.attReprTr)));
attReprTrFull = bsxfun(@rdivide, DATA.attReprTrFull,sqrt(sum(DATA.attReprTrFull.*DATA.attReprTrFull)));
attReprVal = bsxfun(@rdivide, DATA.attReprVa,sqrt(sum(DATA.attReprVa.*DATA.attReprVa)));
attReprTe = bsxfun(@rdivide, DATA.attReprTe,sqrt(sum(DATA.attReprTe.*DATA.attReprTe)));
attReprTr(isnan(attReprTr))=0;
attReprTrFull(isnan(attReprTrFull))=0;

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

stdphocs = std(phocsTrFull,[],2);
stdphocs(stdphocs==0)=1;
phocsTrFull = bsxfun(@rdivide, phocsTrFull,stdphocs);
phocsTr = bsxfun(@rdivide, phocsTr,stdphocs);

% B) For each config, learn on Tr (small), validate on val, and keep the
% best according to QBE map. Other criterions (eg, QBS map or QBS p1) are
% possible.

bestScore = -1;
bestReg = opts.Reg.Reg(1);

for reg=opts.Reg.Reg
    K = size(attReprTr,1);
    Wx = learn_regression(attReprTr', phocsTr',reg);
    
    % Check that there are enough valid projections
    if  ~isreal(Wx(:,1:K))
        continue;
    end
    
    % Embed val and test
    attReprVal_reg = Wx(:,1:K)' * attReprVal;
    phocsVal_reg =  phocsVal;
    attReprTe_reg = Wx(:,1:K)' * attReprTe;
    phocsTe_reg =  phocsTe;
    
    % L2 normalize (critical)
    attReprVal_reg = (bsxfun(@rdivide, attReprVal_reg, sqrt(sum(attReprVal_reg.*attReprVal_reg))));
    attReprTe_reg = (bsxfun(@rdivide, attReprTe_reg, sqrt(sum(attReprTe_reg.*attReprTe_reg))));
    phocsVal_reg = (bsxfun(@rdivide, phocsVal_reg, sqrt(sum(phocsVal_reg.*phocsVal_reg))));
    phocsTe_reg = (bsxfun(@rdivide, phocsTe_reg, sqrt(sum(phocsTe_reg.*phocsTe_reg))));
    
    % Get QBE and QBS scores on validation and test. Note that looking
    % at the test ones is frowned upon. Just for debugging purposes.
    % Evaluate
    % Val
    % QBE
    [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprVal_reg, attReprVal_reg,DATA.wordClsVa,DATA.labelsVa);
    qbe_val_map = mean(mAPEucl);
    qbe_val_p1 = mean(p1);
    
    % QBS (note the 1 at the end)
    [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsVal_reg, attReprVal_reg,DATA.wordClsVa,DATA.labelsVa,1);
    qbs_val_map = mean(mAPEucl);
    qbs_val_p1 = mean(p1);
    
    % Test
    % QBE
    [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_reg, attReprTe_reg,DATA.wordClsTe,DATA.labelsTe);
    qbe_test_map = mean(mAPEucl);
    qbe_test_p1 = mean(p1);
    
    % QBS (note the 1 at the end)
    [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_reg, attReprTe_reg,DATA.wordClsTe,DATA.labelsTe,1);
    qbs_test_map = mean(mAPEucl);
    qbs_test_p1 = mean(p1);
    
    if opts.Reg.verbose
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
        bestReg = reg;
    end
    
end
disp('------------------------------------');
fprintf('Best qbe+qbs map result on validation: %.2f map with reg %.8f\n', 100*bestScore/2, bestReg);
disp('------------------------------------');

%% Part 2: Use the best parameters on the whole thing
fprintf('\n');
disp('**************************************');
disp('***********   Reg FINAL  *************');
disp('**************************************');
% Learn Reg
Wx = learnReg(attReprTrFull', phocsTrFull',bestReg);
K = size(attReprTr,1);
% Embed
% Embed  test
attReprTe_reg = Wx(:,1:K)' * attReprTe;
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
fprintf('\n');
disp('------------------------------------');
fprintf('reg: %.8f. k: %d\n',  bestReg, K);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

mAP = (qbe_test_map+qbs_test_map)/2;
emb_repr.attRepr_emb = attReprTe_reg;
emb_repr.phocRepr_emb = phocsTe_reg;
embedding.Wx = Wx;
embedding.K = K;
embedding.matts = matts;
embedding.mphocs = mphocs;

end