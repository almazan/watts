function  evaluateRegress(  opts,  DATA)
% Evaluate CCA.

%% Part 1: Crosvalidate to find the best parameters in the config range
fprintf('\n');
disp('**************************************');
disp('*************   CV CCA   *************');
disp('**************************************');

% A) L2 normalize and mean center. Not critical, but helps a bit.
attFeatsTr = bsxfun(@rdivide, DATA.attFeatsTr,sqrt(sum(DATA.attFeatsTr.*DATA.attFeatsTr)));
attFeatsTr(isnan(attFeatsTr)) = 0;
attFeatsTrFull = bsxfun(@rdivide, DATA.attFeatsTrFull,sqrt(sum(DATA.attFeatsTrFull.*DATA.attFeatsTrFull)));
attFeatsTrFull(isnan(attFeatsTrFull)) = 0;
attFeatsVal = bsxfun(@rdivide, DATA.attFeatsVal,sqrt(sum(DATA.attFeatsVal.*DATA.attFeatsVal)));
attFeatsVal(isnan(attFeatsVal)) = 0;
attFeatsTe = bsxfun(@rdivide, DATA.attFeatsTe,sqrt(sum(DATA.attFeatsTe.*DATA.attFeatsTe)));
attFeatsTe(isnan(attFeatsTe)) = 0;

hocsTrFull = bsxfun(@rdivide, DATA.hocsTrFull,sqrt(sum(DATA.hocsTrFull.*DATA.hocsTrFull)));
hocsTr = bsxfun(@rdivide, DATA.hocsTr,sqrt(sum(DATA.hocsTr.*DATA.hocsTr)));
hocsVal = bsxfun(@rdivide, DATA.hocsVal,sqrt(sum(DATA.hocsVal.*DATA.hocsVal)));
hocsTe = bsxfun(@rdivide, DATA.hocsTe,sqrt(sum(DATA.hocsTe.*DATA.hocsTe)));

matts = mean(attFeatsTr,2);
attFeatsTrFull = bsxfun(@minus,attFeatsTrFull, matts);
attFeatsTr = bsxfun(@minus,attFeatsTr, matts);
attFeatsVal = bsxfun(@minus,attFeatsVal, matts);
attFeatsTe =  bsxfun(@minus, attFeatsTe,matts);

mhocs = mean(hocsTr,2);
hocsTrFull = bsxfun(@minus, hocsTrFull,mhocs);
hocsTr = bsxfun(@minus, hocsTr,mhocs);
hocsVal = bsxfun(@minus, hocsVal,mhocs);
hocsTe=  bsxfun(@minus, hocsTe,mhocs);

% B) For each config, learn on Tr (small), validate on val, and keep the
% best according to QBE map. Other criterions (eg, QBS map or QBS p1) are
% possible.

bestScore = -1;
bestReg = opts.Regress.Reg(1);

for reg=opts.Regress.Reg
    % Learn CCA using tr (small)
    [Wx,Wy] = regress(attFeatsTr', hocsTr',reg);
        % Check that there are enough valid projections
        if  ~isreal(Wx) || ~isreal(Wy)
            continue;
        end
        
        % Embed val and test
        attFeatsVal_cca = Wx' * attFeatsVal;
        hocsVal_cca = Wy' * hocsVal;
        attFeatsTe_cca = Wx' * attFeatsTe;
        hocsTe_cca = Wy' * hocsTe;
        
        % L2 normalize (critical)
        attFeatsVal_cca = (bsxfun(@rdivide, attFeatsVal_cca, sqrt(sum(attFeatsVal_cca.*attFeatsVal_cca))));
        attFeatsTe_cca = (bsxfun(@rdivide, attFeatsTe_cca, sqrt(sum(attFeatsTe_cca.*attFeatsTe_cca))));
        hocsVal_cca = (bsxfun(@rdivide, hocsVal_cca, sqrt(sum(hocsVal_cca.*hocsVal_cca))));
        hocsTe_cca = (bsxfun(@rdivide, hocsTe_cca, sqrt(sum(hocsTe_cca.*hocsTe_cca))));
        
        % Get QBE and QBS scores on validation and test. Note that looking
        % at the test ones is frowned upon. Just for debugging purposes.
        % Evaluate
        % Val
        % QBE
        [p1,mAPEucl, q] = eval_dp_asymm(opts,attFeatsVal_cca, attFeatsVal_cca,DATA.queriesClassesVal,DATA.wordsVal);
        qbe_val_map = mean(mAPEucl);
        qbe_val_p1 = mean(p1);
        
        % QBS (note the 1 at the end)
        [p1,mAPEucl, q] = eval_dp_asymm(opts,hocsVal_cca, attFeatsVal_cca,DATA.queriesClassesVal,DATA.wordsVal,1);
        qbs_val_map = mean(mAPEucl);
        qbs_val_p1 = mean(p1);
        
        % Test
        % QBE
        [p1,mAPEucl, q] = eval_dp_asymm(opts,attFeatsTe_cca, attFeatsTe_cca,DATA.queriesClassesTe,DATA.wordsTe);
        qbe_test_map = mean(mAPEucl);
        qbe_test_p1 = mean(p1);
        
        % QBS (note the 1 at the end)
        [p1,mAPEucl, q] = eval_dp_asymm(opts,hocsTe_cca, attFeatsTe_cca,DATA.queriesClassesTe,DATA.wordsTe,1);
        qbs_test_map = mean(mAPEucl);
        qbs_test_p1 = mean(p1);
        
        if opts.CCA.verbose
            % Display info
            disp('----------Test results only for debug purposes. Do not use!--------------');
            fprintf('reg: %.8f.\n',  reg);
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
fprintf('Best qbe map result on validation: %.2f map with reg %.8f\n', 100*bestScore, bestReg);
disp('------------------------------------');

%% Part 2: Use the best parameters on the whole thing
fprintf('\n');
disp('**************************************');
disp('***********   CCA FINAL  *************');
disp('**************************************');
% Learn CCA
[Wx,Wy] = regress(attFeatsTrFull', hocsTrFull',bestReg);
% Embed
% Embed  test
attFeatsTe_cca = Wx' * attFeatsTe;
hocsTe_cca = Wy' * hocsTe;

% L2 normalize (critical)
attFeatsTe_cca = (bsxfun(@rdivide, attFeatsTe_cca, sqrt(sum(attFeatsTe_cca.*attFeatsTe_cca))));
hocsTe_cca = (bsxfun(@rdivide, hocsTe_cca, sqrt(sum(hocsTe_cca.*hocsTe_cca))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attFeatsTe_cca, attFeatsTe_cca,DATA.queriesClassesTe,DATA.wordsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,hocsTe_cca, attFeatsTe_cca,DATA.queriesClassesTe,DATA.wordsTe,1);
qbs_test_map = mean(mAPEucl);
qbs_test_p1 = mean(p1);

% Display info
fprintf('\n');
disp('------------------------------------');
fprintf('reg: %.8f. \n',  reg);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

end