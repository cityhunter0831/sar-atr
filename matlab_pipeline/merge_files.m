clear all;close all;clc;
%stuff from main code
fileNamePrefix = {'2S1','BMP2_SN_9563','BMP2_SN_9566','BMP2_SN_C21',...
                'BTR70_SN_C71','T72_SN_132','T72_SN_812','T72_SN_S7','ZSU_23_4'};
%n_samples=[233,233,232,256,299,298,299,299,299,299]
array_dtheta=[-6:0.25:-0.25 0.25:0.25:6];
shiftsy =[0,0.15,0, 0.15];
shiftsx =[0,0, 0.15,0.15];


% Wait for generate_aug_images.m to complete
disp('Waiting for generate_aug_images to complete all samples...');
expected_counts = [24, 11, 11, 10, 24, 8, 8, 8, 32];
while true
    all_done = true;
    for i = 1:9
        class_folder = sprintf('../data/gen_aug_data/%s', fileNamePrefix{i});
        last_file = sprintf('%s/sample_%d.mat', class_folder, expected_counts(i));
        if ~isfile(last_file)
            all_done = false;
            break;
        end
    end
    if all_done
        break;
    end
    pause(10); % Check every 10 seconds
end
disp('All samples found! Starting merge...');

for idxClass = 1:length(fileNamePrefix)
    path2coeff = ('../data/recovered_coefficients/');
    m = load(sprintf('%s%s',path2coeff,fileNamePrefix{idxClass}));
    numTrainingSamples = size(m.y_recovered,3);
    clearvars m
    
    %Merge code
    infer_sample_flag = 0;
    sample_start = 1;
    sample_end = numTrainingSamples;
    class_folder = sprintf('../data/gen_aug_data/%s',fileNamePrefix{idxClass});
    class_file = sprintf('../data/gen_aug_data/%s_aug_images.mat',fileNamePrefix{idxClass});
    s = load(sprintf('%s/sample_%d',class_folder,sample_start));
    factr = size(s.elev,1);
    
    %Initialize empty array
    imgTrain = zeros((length(array_dtheta)+1)*length(shiftsy)*numTrainingSamples,64,64);
    aziTrain = zeros((length(array_dtheta)+1)*length(shiftsy)*numTrainingSamples,1);
    elev = zeros((length(array_dtheta)+1)*length(shiftsy)*numTrainingSamples,1);
    
    %fill the complete matrices
    fprintf('Starting filling class %d from sample %d\n',idxClass,sample_start);
    for idxSample = sample_start:sample_end
        fill_start = 1+(idxSample-1)*factr;
        fill_end = fill_start+(factr-1);
        fprintf('Filling sample %d in array indices %d to %d\n',idxSample,fill_start,fill_end);
        %load and fill
        s = load(sprintf('%s/sample_%d',class_folder,idxSample));
        imgTrain(fill_start:fill_end,:,:) = s.imgTrain;
        aziTrain(fill_start:fill_end,:) = s.aziTrain;
        elev(fill_start:fill_end,:) = s.elev;
    end
    
    %% Save
    fprintf('Saving Data to disk for class %s...\n', fileNamePrefix{idxClass});
    save(class_file,'imgTrain','aziTrain','elev','-v7.3');
end