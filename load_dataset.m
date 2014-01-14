function data = load_dataset(opts)
disp('* Loading dataset *');

if ~exist(opts.fileData, 'file')
    % Loading dataset images and GT
    if strcmpi(opts.dataset,'IAM')
        [data,images] = load_IAM(opts);
    elseif strcmpi(opts.dataset,'IIIT5K')
        [data,images] = load_IIIT5K(opts);
    elseif strcmpi(opts.dataset,'SVT')
        [data,images] = load_SVT(opts);
    elseif strcmpi(opts.dataset,'GW')
        [data,images] = load_GW(opts);
    elseif strcmpi(opts.dataset,'ESP')
        [data,images] = load_ESP(opts);
    end
    
    save(opts.fileData,'data');
    save(opts.fileImages,'images');
else
    load(opts.fileData);
end
    
end