% Preprocess MSTAR 15-degree test data and export to gen_aug_data directory
clear all; clc; close all;

path2mstar = 'C:/Users/kkand/Desktop/25-2 academics/mstar_data_aug/data/mstar_test';
dest_dir = '../data/gen_aug_data';
if ~isfolder(dest_dir)
    mkdir(dest_dir);
end

% Define 5 classes and their variants and extensions
classes = {'2S1', 'BMP2', 'BTR70', 'T72', 'ZSU_23_4'};

variants = { ...
    {'2S1'}, ...
    {'BMP2_SN_9563', 'BMP2_SN_9566', 'BMP2_SN_C21'}, ...
    {'BTR70_SN_C71'}, ...
    {'T72_SN_132', 'T72_SN_812', 'T72_SN_S7'}, ...
    {'ZSU_23_4'} ...
};

extensions = { ...
    {'000'}, ...
    {'000', '001', '002'}, ...
    {'004'}, ...
    {'015', '016', '017'}, ...
    {'026'} ...
};

numPixelsCrop = 128;

for idxClass = 1:length(classes)
    className = classes{idxClass};
    fprintf('Processing Test Class %s ...\n', className);
    
    % Temporary containers for all variants of this class
    all_imgTest = [];
    all_aziTest = [];
    all_elevTest = [];
    
    classVariants = variants{idxClass};
    classExts = extensions{idxClass};
    
    for idxVar = 1:length(classVariants)
        varName = classVariants{idxVar};
        ext = classExts{idxVar};
        
        pathLoad = sprintf('%s/%s/', path2mstar, varName);
        file_names = dir([pathLoad sprintf('*.%s', ext)]);
        file_names = char({file_names.name}');
        
        if isempty(file_names)
            fprintf('  WARNING: No files found for variant %s\n', varName);
            continue;
        end
        
        numFiles = size(file_names, 1);
        fprintf('  Processing variant %s (%d files)...\n', varName, numFiles);
        
        var_img = zeros(numFiles, 64, 64);
        var_azi = zeros(numFiles, 1);
        var_elev = zeros(numFiles, 1);
        
        for i = 1:numFiles
            pathFile = [pathLoad file_names(i,:)];
            gg = MSTAR_LOAD_IMAGE(pathFile);
            
            img_comp = (flipud(gg.ImageData));
            azi = gg.TargetAz;
            dep = gg.MeasuredDepression;
            
            % Square and crop to 128x128 center-crop
            len = min(size(img_comp));
            img_comp = img_comp(1:len, 1:len);
            centerIm = floor(size(img_comp)/2);
            img_comp = img_comp(centerIm(1) - floor(numPixelsCrop/2)+1 : centerIm(1) + floor(numPixelsCrop/2), ...
                                centerIm(2) - floor(numPixelsCrop/2)+1 : centerIm(2) + floor(numPixelsCrop/2));
            
            % Crop the center 64x64 (using coordinates 32:95 consistent with generate_aug_images)
            var_img(i,:,:) = img_comp(32:95, 32:95);
            var_azi(i) = azi;
            var_elev(i) = dep;
        end
        
        all_imgTest = cat(1, all_imgTest, var_img);
        all_aziTest = cat(1, all_aziTest, var_azi);
        all_elevTest = cat(1, all_elevTest, var_elev);
    end
    
    % Save merged class test file
    imgTest = all_imgTest;
    aziTest = all_aziTest;
    elev = all_elevTest;
    
    save(sprintf('%s/%s_test.mat', dest_dir, className), 'imgTest', 'aziTest', 'elev');
    fprintf('  Saved merged test set to %s/%s_test.mat (Total %d samples)\n', dest_dir, className, length(aziTest));
end

fprintf('\nAll test set exporting complete!\n');
