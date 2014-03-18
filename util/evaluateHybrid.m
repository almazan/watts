function hybrid_map = evaluateHybrid(opts,DATA,embedding)

fprintf('\n');
disp('**************************************');
disp('************  Hybrid KCSR  ***********');
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
alpha = 0:0.1:1;
hybrid_map = zeros(length(alpha),1);
hybrid_p1 = zeros(length(alpha),1);
for i=1:length(alpha)
    attRepr_hybrid = attReprTe_cca*alpha(i) + phocsTe_cca*(1-alpha(i));
    [p1,mAPEucl,q] = eval_dp_asymm(opts,attRepr_hybrid,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
    hybrid_map(i) = mean(mAPEucl)*100;
    hybrid_p1(i) = mean(p1)*100;
end

[best_map,idx] = max(hybrid_map);
best_p1 = hybrid_p1(idx);
best_alpha = alpha(idx);

% Display info
disp('------------------------------------');
fprintf('alpha: %.2f reg: %.8f. k: %d\n', best_alpha, embedding.reg, embedding.K);
fprintf('hybrid --   test: (map: %.2f. p@1: %.2f)\n',  best_map, best_p1);
disp('------------------------------------');

plot(alpha,hybrid_map,'.-','MarkerSize',16)
title(opts.dataset)
xlabel('alpha')
ylabel('Mean Average Precision (%)')
grid on

end
