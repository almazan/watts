function  [embedding,emb_repr,mAP] = evaluateKCCA(opts,DATA)
% Evaluate KCCA.

%% Part 1: Crosvalidate to find the best parameters in the config range
fprintf('\n');
disp('**************************************');
disp('*************   CV KCCA   ************');
disp('**************************************');


% For each config, learn on Tr (small), validate on val, and keep the
% best according to QBE+QBS map. Other criterions (eg, QBE map or QBS p1) are
% possible.

bestScore = -1;
bestReg = opts.KCCA.Reg(end);
bestK = opts.KCCA.Dims(end);
bestG = opts.KCCA.G(end);
bestM = opts.KCCA.M(end);

D = size(DATA.attReprTr,1);
for G=opts.KCCA.G
    % Create random projections matrix
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',0));
    rndmat = normrnd(0,1/G, 2500,D);
    for M=opts.KCCA.M
        % Cut slice
        mat = rndmat(1:M,:);
        
        % Project attributes and phocs using the explicit exponential
        % embedding. Project train, val, and test.
        % train
        tmp = mat*DATA.attReprTr;
        attReprTr_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        tmp = mat*DATA.phocsTr;
        phocsTr_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        % val
        tmp = mat*DATA.attReprVa;
        attReprVal_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        tmp = mat*DATA.phocsVa;
        phocsVal_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        % test
        tmp = mat*DATA.attReprTe;
        attReprTe_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        tmp = mat*DATA.phocsTe;
        phocsTe_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        
        % Mean center
        ma = mean(attReprTr_emb,2);
        attReprTr_emb=bsxfun(@minus, attReprTr_emb, ma);
        attReprVal_emb=bsxfun(@minus, attReprVal_emb, ma);
        attReprTe_emb=bsxfun(@minus, attReprTe_emb, ma);
        mh = mean(phocsTr_emb,2);
        phocsTr_emb=bsxfun(@minus, phocsTr_emb, mh);
        phocsVal_emb=bsxfun(@minus, phocsVal_emb, mh);
        phocsTe_emb=bsxfun(@minus, phocsTe_emb, mh);
        
        % Learn (K)CCA
        for reg=opts.KCCA.Reg
            % Learn CCA using tr (small)
            [Wx,Wy,r] = cca2(attReprTr_emb', phocsTr_emb',reg,max(opts.KCCA.Dims));
            for K=opts.KCCA.Dims
                % Check that there are enough valid projections
                if  ~isreal(Wx(:,1:K)) || ~isreal(Wy(:,1:K))
                    continue;
                end
                
                % Embed val and test
                attReprVal_cca = Wx(:,1:K)' * attReprVal_emb;
                phocsVal_cca = Wy(:,1:K)' * phocsVal_emb;
                attReprTe_cca = Wx(:,1:K)' * attReprTe_emb;
                phocsTe_cca = Wy(:,1:K)' * phocsTe_emb;
                
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
                
                if opts.KCCA.verbose
                    % Display info
                    disp('----------Test results only for debug purposes. Do not use!--------------');
                    fprintf('reg: %.8f. k: %d, M: %d, G: %d\n',  reg, K, M, G);
                    fprintf('qbe -- val: (map: %.2f. p@1: %.2f).  test: (map: %.2f. p@1: %.2f)\n', 100*qbe_val_map, 100*qbe_val_p1, 100*qbe_test_map, 100*qbe_test_p1);
                    fprintf('qbs -- val: (map: %.2f. p@1: %.2f).  test: (map: %.2f. p@1: %.2f)\n', 100*qbs_val_map, 100*qbs_val_p1, 100*qbs_test_map, 100*qbs_test_p1);
                    disp('------------------------------------');
                end
                
                % Better than before? update.
                if qbe_val_map + qbs_val_map > bestScore
                    bestScore = qbe_val_map + qbs_val_map;
                    bestK = K;
                    bestReg = reg;
                    bestM = M;
                    bestG = G;
                end
            end
        end
        
    end
end



disp('------------------------------------');
fprintf('Best qbe map result on validation: %.2f map with reg %.8f, K %d, G %d, and M %d\n', 100*bestScore, bestReg, bestK, bestG, bestM);
disp('------------------------------------');

%% Part 2: Use the best parameters on the whole thing
fprintf('\n');
disp('**************************************');
disp('***********   KCCA FINAL  ************');
disp('**************************************');

% Get random matrix
RandStream.setGlobalStream(RandStream('mt19937ar','seed',0));
rndmat = normrnd(0,1/bestG, 2500,D);
mat = rndmat(1:bestM,:);
% Embed train (full) and test
% Project attributes and phocs using the explicit exponential
% embedding. Project train, val, and test.
% train
tmp = mat*DATA.attReprTrFull;
attReprTrFull_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
tmp = mat*DATA.phocsTrFull;
phocsTrFull_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
% test
tmp = mat*DATA.attReprTe;
attReprTe_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
tmp = mat*DATA.phocsTe;
phocsTe_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];

% Mean center
ma = mean(attReprTrFull_emb,2);
attReprTrFull_emb=bsxfun(@minus, attReprTrFull_emb, ma);
attReprTe_emb=bsxfun(@minus, attReprTe_emb, ma);

mh = mean(phocsTrFull_emb,2);
phocsTrFull_emb=bsxfun(@minus, phocsTrFull_emb, mh);
phocsTe_emb=bsxfun(@minus, phocsTe_emb, mh);

% Learn CCA
[Wx,Wy,r] = cca2(attReprTrFull_emb', phocsTrFull_emb',bestReg,bestK);

% Embed test
attReprTe_cca = Wx(:,1:bestK)' * attReprTe_emb;
phocsTe_cca = Wy(:,1:bestK)' * phocsTe_emb;
attReprTrFull_cca = Wx(:,1:bestK)' * attReprTrFull_emb;


% L2 normalize (critical)
attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));
attReprTrFull_cca = (bsxfun(@rdivide, attReprTrFull_cca, sqrt(sum(attReprTrFull_cca.*attReprTrFull_cca))));

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
embedding.K = bestK;
embedding.M = bestM;
embedding.reg = bestReg;
embedding.rndmat = rndmat;
embedding.matts = ma;
embedding.mphocs = mh;

end