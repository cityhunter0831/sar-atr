clc;
clear all;
close all;

addpath('spgl1-2.1');
addpath('mtimesx');

fileNamePrefix = {'2S1','BMP2_SN_9563','BMP2_SN_9566','BMP2_SN_C21',...
                'BTR70_SN_C71','T72_SN_132','T72_SN_812','T72_SN_S7','ZSU_23_4'};
gaussWidth = 1;

%% Iterate over all classes

for idxClass =1:length(fileNamePrefix)
    data_PH=load(sprintf('../data/phase_histories/%s_PH',fileNamePrefix{idxClass}));
    numImages = size(data_PH.arr_img_fft_polar,3);
    f_center = 9.6e9;
    bandwidth = 521e6;
    delF = bandwidth/100;
    fLower = f_center - bandwidth/2;
    f = linspace(fLower,fLower + bandwidth,100 ).';
    
    % Using the horizontal polarization measurements
    thetas = (-1.5:0.03:1.5-0.03);
    bisectorAngles = 90 + thetas;
    azimuthVals = bisectorAngles;
    deltaAngles = 0;
    numChips = size(data_PH.arr_img_fft_polar,3);   
    velLight = 299792458;
    
    numFreqBins = length(f);
    pixelResolutionMSTAR = 0.202;
    numRangeBins = numFreqBins; % Number of range bins are calculated from the range resolution
    
    numAzimuthBinsTotal = length(bisectorAngles); % Number of azimuth looks are calculated across the range [0, 3) with resolution dependent on cross-range extent
    AzimuthBasisCenterSpacing = 0.4; % Degrees
    numAzimuthBasisCenters = round(3/AzimuthBasisCenterSpacing);
    
    gaussWidthMax=5;
    gaussWidthMin = 1;
    
    %The angular and spatial grid points
    L =30;
    azimuthBasisCenters = 90+ linspace(-1.5,1.5,numAzimuthBasisCenters);

    %using range resolution=0.3
    xGrids = -L/2:0.3:L/2-0.3;
    yGrids = -L/2:0.3:L/2-0.3;
    
    [X,Y] = meshgrid(xGrids,yGrids);
    Xp = repmat(X(:)',numFreqBins,1);
    Yp = repmat(Y(:)',numFreqBins,1);
    X = X(:);
    Y = Y(:);
    F = repmat(f,1,numRangeBins^2);
    
    BasisFuncTemp=[];
    groups_l12 = [];
    
    dist = pdist2(azimuthVals.',azimuthBasisCenters');
    BasisFunc = exp(-0.5*dist.^2/gaussWidth^2);
    normBasisFunc = diag((sum(BasisFunc.^2,1)).^0.5);
    BasisFunc = BasisFunc/normBasisFunc;
    numVariables= size(BasisFunc,2);
    groups_l12 = repmat((1:numRangeBins^2).',1,numVariables);
    groups_l12=groups_l12(:);
    
    numPulses = numAzimuthBinsTotal;
    pulseSelectIDx = 1:length(azimuthVals);
    az =azimuthVals;
    
    
    % Few-shot random sampling: Select random chips matching the paper's distribution
    % 2S1: 24, BMP2: 32 (11/11/10), BTR70: 24, T72: 24 (8/8/8), ZSU23: 32 (Total: 136)
    rng(42); % Fixed seed for reproducibility
    switch fileNamePrefix{idxClass}
        case '2S1'
            numTargetChips = 24;
        case 'BMP2_SN_9563'
            numTargetChips = 11;
        case 'BMP2_SN_9566'
            numTargetChips = 11;
        case 'BMP2_SN_C21'
            numTargetChips = 10;
        case 'BTR70_SN_C71'
            numTargetChips = 24;
        case 'T72_SN_132'
            numTargetChips = 8;
        case 'T72_SN_812'
            numTargetChips = 8;
        case 'T72_SN_S7'
            numTargetChips = 8;
        case 'ZSU_23_4'
            numTargetChips = 32;
        otherwise
            numTargetChips = 15;
    end
    selected_indices = sort(randperm(numChips, min(numTargetChips, numChips)));
    numChipsSelected = length(selected_indices);
    
    y_recovered = zeros(100,100,numChipsSelected);
    y_residual =  zeros(100,100,numChipsSelected);
    x_recovered = zeros(100*100*numVariables,numChipsSelected);
    fileName = sprintf('../data/recovered_coefficients/%s',fileNamePrefix{idxClass});
    gaussWidthStore=zeros(numChipsSelected,1);
    
    % Limit threads on main process to prevent oversubscription
    maxNumCompThreads(1);
    
    parfor idxChipsSelected = 1:numChipsSelected
        idxChips = selected_indices(idxChipsSelected);
        fprintf('processing class=%d, image_idx=%d (%d/%d)\n', idxClass, idxChips, idxChipsSelected, numChipsSelected);
        
        % Limit threads inside parfor worker process
        mtimesx('NUM_THREADS', 1);
        
        depression = data_PH.depression(idxChips);
        A_mod = zeros(numFreqBins,numRangeBins^2,numAzimuthBinsTotal);
        for idxPulses=1:numAzimuthBinsTotal
            A_mod(:,:,idxPulses) =1/sqrt(numFreqBins)*exp(1i*4*pi*F*...
                cosd(depression)/velLight.*(Xp.*cosd(azimuthVals(idxPulses)) +...
                Yp.*sind(azimuthVals(idxPulses)) ));
        end
        
        
        y1= (data_PH.arr_img_fft_polar(:,:,idxChips));
        y1=y1(:); 
        snr = 20;
        numSamples = length(y1);
        sigma_n = sqrt(2)*norm(y1)*10^(-snr/20);
        
        A_mod_sliced = A_mod(:,:,pulseSelectIDx);
        A = @(x,mode) SAR_operator_gen(x,mode,pulseSelectIDx,numRangeBins,numFreqBins,azimuthVals,BasisFunc,A_mod_sliced);
        opts = spgSetParms('iscomplex',1,'verbosity',0,'iterations',100);
        %Find Optimum C
        C = spg_group(A, y1, groups_l12, sigma_n, opts );
        
        %Get Derived quantities
        x_recovered(:,idxChipsSelected) = C;
        y_recovered(:,:,idxChipsSelected) = reshape(A(C,1),100,100);
        y_residual(:,:,idxChipsSelected) = fliplr(data_PH.arr_img_fft_polar(:,:,idxChips)) - y_recovered(:,:,idxChipsSelected);
        gaussWidthStore(idxChipsSelected) = 1.0;
    end
    save(fileName,'x_recovered','y_recovered','y_residual','gaussWidthStore','selected_indices');
end