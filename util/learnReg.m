function  [embedding,mAP] = learnReg(opts,DATA)
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
attReprTr(isnan(attReprTr))=0;
attReprTrFull(isnan(attReprTrFull))=0;

phocsTrFull = bsxfun(@rdivide, DATA.phocsTrFull,sqrt(sum(DATA.phocsTrFull.*DATA.phocsTrFull)));
phocsTr = bsxfun(@rdivide, DATA.phocsTr,sqrt(sum(DATA.phocsTr.*DATA.phocsTr)));
phocsVal = bsxfun(@rdivide, DATA.phocsVa,sqrt(sum(DATA.phocsVa.*DATA.phocsVa)));

matts = mean(attReprTr,2);
attReprTrFull = bsxfun(@minus,attReprTrFull, matts);
attReprTr = bsxfun(@minus,attReprTr, matts);
attReprVal = bsxfun(@minus,attReprVal, matts);

mphocs = mean(phocsTr,2);
phocsTrFull = bsxfun(@minus, phocsTrFull,mphocs);
phocsTr = bsxfun(@minus, phocsTr,mphocs);
phocsVal = bsxfun(@minus, phocsVal,mphocs);

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
    
    % Embed val
    attReprVal_reg = Wx(:,1:K)' * attReprVal;
    phocsVal_reg =  phocsVal;
    
    % L2 normalize (critical)
    attReprVal_reg = (bsxfun(@rdivide, attReprVal_reg, sqrt(sum(attReprVal_reg.*attReprVal_reg))));
    phocsVal_reg = (bsxfun(@rdivide, phocsVal_reg, sqrt(sum(phocsVal_reg.*phocsVal_reg))));
    
    % Get QBE and QBS scores on validation 
    % QBE
    [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprVal_reg, attReprVal_reg,DATA.wordClsVa,DATA.labelsVa);
    qbe_val_map = mean(mAPEucl);
    qbe_val_p1 = mean(p1);
    
    % QBS (note the 1 at the end)
    [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsVal_reg, attReprVal_reg,DATA.wordClsVa,DATA.labelsVa,1);
    qbs_val_map = mean(mAPEucl);
    qbs_val_p1 = mean(p1);

    
    if opts.Reg.verbose
        % Display info
        disp('------------------------------------');
        fprintf('reg: %.8f. k: %d\n',  reg, K);
        fprintf('qbe -- val: (map: %.2f. p@1: %.2f)\n', 100*qbe_val_map, 100*qbe_val_p1);
        fprintf('qbs -- val: (map: %.2f. p@1: %.2f)\n', 100*qbs_val_map, 100*qbs_val_p1);
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

% Learn Reg
Wx = learn_regression(attReprTrFull', phocsTrFull',bestReg);
K = size(attReprTr,1);

embedding.Wx = Wx;
embedding.K = K;
embedding.matts = matts;
embedding.mphocs = mphocs;
embedding.reg = bestReg;
mAP = 100*bestScore/2;

end