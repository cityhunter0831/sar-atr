% Export baseline (136 raw training chips without augmentation) to gen_aug_data directory
clear all; clc; close all;

path2PH = '../data/phase_histories/';
dest_dir = '../data/gen_aug_data';
if ~isfolder(dest_dir)
    mkdir(dest_dir);
end

% Define 5 classes and their training variants
classes = {'2S1', 'BMP2', 'BTR70', 'T72', 'ZSU_23_4'};

variants = { ...
    {'2S1'}, ...
    {'BMP2_SN_9563', 'BMP2_SN_9566', 'BMP2_SN_C21'}, ...
    {'BTR70_SN_C71'}, ...
    {'T72_SN_132', 'T72_SN_812', 'T72_SN_S7'}, ...
    {'ZSU_23_4'} ...
};

% Match sparse_recovery's exact few-shot indices (seeded by rng(42))
% 2S1: 24, BMP2: 32 (11/11/10), BTR70: 24, T72: 24 (8/8/8), ZSU23: 32 (Total: 136)
numTargetChips_map = containers.Map();
numTargetChips_map('2S1') = 24;
numTargetChips_map('BMP2_SN_9563') = 11;
numTargetChips_map('BMP2_SN_9566') = 11;
numTargetChips_map('BMP2_SN_C21') = 10;
numTargetChips_map('BTR70_SN_C71') = 24;
numTargetChips_map('T72_SN_132') = 8;
numTargetChips_map('T72_SN_812') = 8;
numTargetChips_map('T72_SN_S7') = 8;
numTargetChips_map('ZSU_23_4') = 32;

for idxClass = 1:length(classes)
    className = classes{idxClass};
    fprintf('Exporting Baseline Class %s ...\n', className);
    
    all_imgTrain = [];
    all_aziTrain = [];
    all_elev = [];
    
    classVariants = variants{idxClass};
    
    for idxVar = 1:length(classVariants)
        varName = classVariants{idxVar};
        
        % Load phase history file (contains arr_img_comp)
        PH = load(sprintf('%s%s_PH.mat', path2PH, varName));
        numChips = size(PH.arr_img_comp, 1);
        
        % Sample the exact same indices as sparse_recovery
        rng(42);
        numTarget = numTargetChips_map(varName);
        selected_indices = sort(randperm(numChips, min(numTarget, numChips)));
        
        fprintf('  Variant %s: selected %d chips...\n', varName, length(selected_indices));
        
        for i = 1:length(selected_indices)
            idx = selected_indices(i);
            % Crop raw complex image to 64x64 (coordinates 32:95)
            raw_img = squeeze(PH.arr_img_comp(idx, :, :));
            cropped_img = raw_img(32:95, 32:95);
            
            all_imgTrain = cat(1, all_imgTrain, reshape(cropped_img, 1, 64, 64));
            all_aziTrain = cat(1, all_aziTrain, PH.arr_azi(idx));
            all_elev = cat(1, all_elev, PH.depression(idx));
        end
    end
    
    imgTrain = all_imgTrain;
    aziTrain = all_aziTrain;
    elev = all_elev;
    
    save(sprintf('%s/%s_baseline.mat', dest_dir, className), 'imgTrain', 'aziTrain', 'elev');
    fprintf('  Saved baseline to %s/%s_baseline.mat (Total %d samples)\n', dest_dir, className, length(aziTrain));
end

fprintf('\nAll baseline exporting complete!\n');
