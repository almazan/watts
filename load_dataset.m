function data = load_dataset(opts)
disp('* Loading dataset *');

if ~exist(opts.fileData, 'file')
    % Loading dataset images and GT
    if strcmpi(opts.dataset,'IAM')        
        data = load_IAM(opts);
    elseif strcmpi(opts.dataset,'IIIT5K')
        data = load_IIIT5K(opts);
    elseif strcmpi(opts.dataset,'SVT')        
        data = load_SVT(opts);
    elseif strcmpi(opts.dataset,'ICDAR11')
        error('Not working');
        [data,images] = load_ICDAR11(opts);
    elseif strcmpi(opts.dataset,'GW')
        error('Not working');
        [data,images] = load_GW(opts);
    elseif strcmpi(opts.dataset,'ESP')
        error('Not working');
        [data,images] = load_ESP(opts);    
    elseif strcmpi(opts.dataset,'LP')        
        data = load_LP(opts);
    end
    save(opts.fileData,'data');
else
    load(opts.fileData);
end
    
end