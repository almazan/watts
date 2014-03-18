function  [embedding,mAP] = learnKCCA(opts,DATA)
% Evaluate KCCA.

%% Part 1: Crosvalidate to find the best parameters in the config range
fprintf('\n');
disp('**************************************');
disp('*************   CV KCSR   ************');
disp('**************************************');


% For each config, learn on Tr (small), validate on val, and keep the
% best according to QBE+QBS map. Other criterions (eg, QBE map or QBS p1) are
% possible.

bestScore = -1;
bestReg = opts.KCCA.Reg(end);
bestK = opts.KCCA.Dims(end);
bestG = opts.KCCA.G(end);
bestM = opts.KCCA.M(end);

Dx = size(DATA.attReprTr,1);
Dy = size(DATA.phocsTr,1);
for G=opts.KCCA.G
    % Create random projections matrix
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',0));
    rndmatx = normrnd(0,1/G, 2500,Dx);
    rndmaty = normrnd(0,1/G, 2500,Dy);
    

    for M=opts.KCCA.M
        % Cut slice
        matx = rndmatx(1:M,:);
        maty = rndmaty(1:M,:);
        
        % Project attributes and phocs using the explicit exponential
        % embedding. Project train, val, and test.
        % train
        tmp = matx*DATA.attReprTr;
        attReprTr_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        tmp = maty*DATA.phocsTr;
        phocsTr_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        % val
        tmp = matx*DATA.attReprVa;
        attReprVal_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        tmp = maty*DATA.phocsVa;
        phocsVal_emb = 1/sqrt(M) * [ cos(tmp); sin(tmp)];
        
        % Mean center
        ma = mean(attReprTr_emb,2);
        attReprTr_emb=bsxfun(@minus, attReprTr_emb, ma);
        attReprVal_emb=bsxfun(@minus, attReprVal_emb, ma);
        mh = mean(phocsTr_emb,2);
        phocsTr_emb=bsxfun(@minus, phocsTr_emb, mh);
        phocsVal_emb=bsxfun(@minus, phocsVal_emb, mh);
        
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
                
                % L2 normalize (critical)
                attReprVal_cca = (bsxfun(@rdivide, attReprVal_cca, sqrt(sum(attReprVal_cca.*attReprVal_cca))));
                phocsVal_cca = (bsxfun(@rdivide, phocsVal_cca, sqrt(sum(phocsVal_cca.*phocsVal_cca))));
                
                % Get QBE and QBS scores on validation
                % QBE
                [p1,mAPEucl, q] = eval_dp_asymm(opts,attReprVal_cca, attReprVal_cca,DATA.wordClsVa,DATA.labelsVa);
                qbe_val_map = mean(mAPEucl);
                qbe_val_p1 = mean(p1);
                
                % QBS (note the 1 at the end)
                [p1,mAPEucl, q] = eval_dp_asymm(opts,phocsVal_cca, attReprVal_cca,DATA.wordClsVa,DATA.labelsVa,1);
                qbs_val_map = mean(mAPEucl);
                qbs_val_p1 = mean(p1);
                
                if opts.KCCA.verbose
                    % Display info
                    disp('------------------------------------');
                    fprintf('reg: %.8f. k: %d, M: %d, G: %d\n',  reg, K, M, G);
                    fprintf('qbe -- val: (map: %.2f. p@1: %.2f)\n', 100*qbe_val_map, 100*qbe_val_p1);
                    fprintf('qbs -- val: (map: %.2f. p@1: %.2f)\n', 100*qbs_val_map, 100*qbs_val_p1);
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
fprintf('Best qbe map result on validation: %.2f map with reg %.8f, K %d, G %d, and M %d\n', 100*bestScore/2, bestReg, bestK, bestG, bestM);
disp('------------------------------------');

% Get random matrix
RandStream.setGlobalStream(RandStream('mt19937ar','seed',0));
rndmatx = normrnd(0,1/bestG, 2500,Dx);
matx = rndmatx(1:bestM,:);
rndmaty = normrnd(0,1/bestG, 2500,Dy);
maty = rndmaty(1:bestM,:);
% Embed train (full) and test
% Project attributes and phocs using the explicit exponential
% embedding. Project train, val, and test.
% train
tmp = matx*DATA.attReprTrFull;
attReprTrFull_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];
tmp = maty*DATA.phocsTrFull;
phocsTrFull_emb = 1/sqrt(bestM) * [ cos(tmp); sin(tmp)];

% Mean center
ma = mean(attReprTrFull_emb,2);
attReprTrFull_emb=bsxfun(@minus, attReprTrFull_emb, ma);

mh = mean(phocsTrFull_emb,2);
phocsTrFull_emb=bsxfun(@minus, phocsTrFull_emb, mh);

% Learn CCA
[Wx,Wy,r] = cca2(attReprTrFull_emb', phocsTrFull_emb',bestReg,bestK);

mAP = 100*bestScore/2;
embedding.Wx = Wx;
embedding.Wy = Wy;
embedding.K = bestK;
embedding.M = bestM;
embedding.reg = bestReg;
embedding.rndmatx = rndmatx;
embedding.rndmaty = rndmaty;
embedding.matts = ma;
embedding.mphocs = mh;

end