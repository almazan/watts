function platts = learnPlatts(  opts,  DATA)
%% Platts
fprintf('\n');
disp('**************************************');
disp('*************   PLATTS   *************');
disp('**************************************');

%% Calibrate using the training set
platts = zeros(size(DATA.attReprTr,1),2);
% Learn
for i=1:size(DATA.attReprTr,1)
    if opts.Platts.verbose && mod(i,50)==0
        fprintf('Learning to calibrate %d/%d\n',i,size(DATA.attReprTr,1));
    end
    platts(i,:) = calib_c(double(DATA.attReprTrFull(i,:)), double(DATA.phocsTrFull(i,:)~=0));
end

end

