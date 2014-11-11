addpath('../util');
addpath('../util/io');

dataset = 'SVT';

evalReco = 0;
evalQBS = 1;

fid = fopen('results_SVT_qbs.txt','a');

for numversion=1:1
    %numversion = 4;
    if numversion == 1
        version = '';
    else
        version = sprintf('v_%d',numversion);
    end
    subset = sprintf('%s%s',dataset,version);
    pathData = 'data/';
    pathDataset = [pathData dataset];
    embeddingMethod = 'cca';
    
    load(fullfile(pathData,sprintf('%s_testdata.mat',dataset)));
    load(fullfile(pathData,sprintf('%s_lexicon_PHOCs_l2345_lb2_nb50.mat',dataset)));
    load(fullfile(pathData,sprintf('%s_wordsGT.mat',dataset)));
    phocs = lexicon.phocs;
    
    % Embed PHOCs
    if strcmpi(embeddingMethod,'cca')
        
        embedding = readCCA(fullfile(pathDataset,'CCA.bin'));
        phocs = bsxfun(@rdivide, phocs,sqrt(sum(phocs.*phocs)));
        phocs=  bsxfun(@minus, phocs,embedding.mphocs);
        phocs = embedding.Wy(:,1:embedding.K)' * phocs;
        phocs = (bsxfun(@rdivide, phocs, sqrt(sum(phocs.*phocs))));
        phocs(isnan(phocs)) = 0;
        
    elseif strcmpi(embeddingMethod,'kcca')
        embedding = readKCCA(fullfile(pathDataset,'KCCA.bin'));
        
        mat = embedding.rndmat(1:embedding.M,:);
        tmp = mat*phocs;
        phocs = 1/sqrt(embedding.M) * [ cos(tmp); sin(tmp)];
        % Mean center
        phocs=bsxfun(@minus, phocs, embedding.mphocs);
        % Embed test
        phocs = embedding.Wy(:,1:embedding.K)' * phocs;
        % L2 normalize (critical)
        phocs = (bsxfun(@rdivide, phocs, sqrt(sum(phocs.*phocs))));
        phocs(isnan(phocs)) = 0;
    end
    
    %if ~exist('candidates','var')
        file = fullfile(pathData,sprintf('%s_candidates.mat',subset));
        if ~exist(file,'file')
            [candidates,docsDic] = read_candidate_regions(sprintf('data/cropped_boxes/experiment_th/%s/%s/',version,dataset),wordsGT,docsDic);
            save(file,'candidates','docsDic','-v7.3');
        else
            load(file)
        end
    %end
    
    file = fullfile(pathData,sprintf('%s_attRepr.bin',subset));
    if ~exist(file,'file')
        ncandidates = length(candidates);
        imagesPerBatch = 1024;
        atts = zeros(embedding.K,ncandidates,'single');
        nbatches = ceil(ncandidates/imagesPerBatch);
        for i = 1:nbatches
            fprintf('Processing batch %d/%d\n',i,nbatches);
            idx = 1 + (i-1)*imagesPerBatch;
            iniIdx = idx;
            endIdx = min(idx+imagesPerBatch-1,ncandidates);
            images = {candidates(iniIdx:endIdx).im};
            att = image2att(images,pathDataset,embeddingMethod);
            atts(:,iniIdx:endIdx) = att;
        end
        writeMat(atts,file);
    else
        atts = readMat(file);
    end
    
    locCand = [candidates(:).x1; candidates(:).x2; ...
        candidates(:).y1; candidates(:).y2; candidates(:).imId]';
    
    if evalReco
        fullLex = {};
        for i=1:length(test)
            fullLex = [fullLex {test(i).words.tag}];
        end
        fullLex = unique(fullLex);
        
        ids = [candidates(:).docId];
        results = [];
        resScores = [];
        resCandidates = [];
        nRelevants = length(wordsGT);
        found = zeros(nRelevants,1);
        foundScores = ones(nRelevants,1)*-1;
        draw = 0;
        path = 'qualres/';
        totalCand = 0;
        totalCorrect = 0;
        rng(0);
        for i=1:length(test)
            fprintf('Evaluating image %d\n',i);
            t = test(i);
            words = t.words;
            nw = length(words);
            locW = [words(:).bb];
            locW = [locW(:).x locW(:).y locW(:).width locW(:).height];
            locW = reshape(locW,length(words),4);
            gtw = {words(:).tag};
            if strcmpi(dataset,'ICDAR03')
                idxrand = randperm(length(fullLex),50);
                lex = [gtw fullLex(idxrand)];
                lex = [gtw fullLex];
            else
                lex = t.lex;
            end
            %     idxw = find(ismember(lexicon.words,gtw));
            idxl = zeros(length(lex),1);
            for j=1:length(lex)
                idxl(j) = find(ismember(lexicon.words,lex{j}));
            end
            ph = phocs(:,idxl);
            if strcmpi(dataset,'ICDAR03')
                nameDoc = t.name;
                idxs = find(nameDoc=='/');
                nameDoc = nameDoc(idxs(1)+1:end);
                nameDoc = nameDoc(1:end-4);
            else
                nameDoc = t.name(5:9);
            end
            if docsDic.isKey(nameDoc)
                idDoc = docsDic(nameDoc);
            else
                continue;
            end
            idxc = find(ids==idDoc);
            attsTemp = atts(:,idxc);
            loc = locCand(idxc,:);
            S = attsTemp'*ph;
            [s,I] = max(S,[],2);
            [s2,I2] = sort(s);
            %     idxs = s2>0.45;
            %     s2 = s2(idxs);
            %     I2 = I2(idxs);
            pick = nms_C(int32(I2),int32(loc)',0.2);
            resL = I(pick);
            resC = idxc(pick);
            resS = s(pick);
            resLoc = locCand(resC,:);
            resCand = candidates(resC);
            
            if draw
                h=figure;imshow(fullfile('datasets/SVT/',t.name));
                for j=1:length(words)
                    rectangle('Position',locW(j,:),'EdgeColor','green');
                end
                for j=1:min(length(pick),10)
                    rectangle('Position',[resCand(j).x1,resCand(j).y1,resCand(j).w,resCand(j).h],'EdgeColor','red');
                    text(double(resCand(j).x1),double(resCand(j).y1)-10,sprintf('%s - %d',lex{resL(j)},j),'Color','red')
                end
                ginput();
                close(h);
            end
            
            totalCand = totalCand + length(resCand);
            resScores = [resScores; resS];
            resCandidates = [resCandidates; resC'];
            for j=1:length(pick)
                if strcmpi(lex{resL(j)},resCand(j).gttext) && resCand(j).wId>0 && found(resCand(j).wId)==0
                    found(resCand(j).wId) = 1;
                    foundScores(resCand(j).wId) = resS(j);
                    results = [results; 1];
                else
                    results = [results; -1];
                end
            end
            
            
            %     intArea = rectint(single(locCand), single(locW));
            %     areaP = single(candidates(i).h*candidates(i).w);
            %     areaGt = single(([wordsGT(:).h].*[wordsGT(:).w])');
            %
            %     denom = bsxfun(@minus, single(areaP+areaGt'), intArea);
            %     overlap = intArea./denom;
            %
            %     [y,x] = find(overlap >= 0.5 & imIds==candidates(i).imId);
            
            
            %     res = [s(idxs) I(idxs)];
            %     res = sort(res,'descend');
            
            %     for j=1:length(I)
            %         imwrite(candidates(idxc(j)).im,sprintf('%s%d_%s.jpg',path,j,lex{I(j)}));
            %     end
        end
        
        
        [resScores,Is] = sort(resScores,'descend');
        results = results(Is);
        resCandidates = resCandidates(Is);
        [rec,prec] = vl_pr(results,resScores,'NumPositives',nRelevants);
        
        % recall = sum(results>0)/nRelevants;
        % prec = sum(results>0)/totalCand;
        [fmeasure,idx] = max(2*(prec.*rec./(prec+rec)));
        recall = rec(idx);
        precision = prec(idx);
        mAP = compute_mAP(results,nRelevants);
        fprintf('mAP: %f\nMax Recall: %f\nRecall: %f\nPrecision: %f\nF-measure: %f\nNum candidates: %d\n',mAP,max(rec),recall,precision,fmeasure,length(candidates));
        fprintf(fid,'%d\n%f\n%f\n%f\n%f\n\n',length(candidates),recall*100,precision*100,fmeasure*100,max(rec)*100);
    end
    
    if evalQBS
    % spotting
    queries = unique({wordsGT(:).gttext});
    idx = find(ismember(lexicon.words,queries));
    qphocs = phocs(:,idx);
    imIds =[candidates(:).imId];
    
    APs = zeros(length(queries),1);
    parfor i=1:length(queries)
        %fprintf('Evaluating query %d\n',i);
        q = queries{i};
        sc = qphocs(:,i)'*atts;
        
        relevant = zeros(length(test),1);
        scores = zeros(length(test),1);
        for j=1:length(test)
            idx = imIds==j;
            if sum(idx)~=0
                scores(j)=max(sc(imIds==j));
            else
                scores(j)=-1;
            end
            words = {test(j).words(:).tag};
            if find(ismember(words,q))
                relevant(j) = 1;
            end
        end
        
        %[sc,Is] = sort(sc);
        %pick = nms_C(int32(Is)',int32(locCand)',0.2);
        [scores,I] = sort(scores,'descend');
        relevant = relevant(I);
        nRelevants = sum(relevant);
        APs(i) = compute_mAP(relevant,nRelevants);
    end
    fprintf(fid,'%d\n%f\n\n',length(candidates),mean(APs));
    end
end
fclose(fid);