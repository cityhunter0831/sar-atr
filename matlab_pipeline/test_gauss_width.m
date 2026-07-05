% Test different gaussWidth (sigma_G) values on representative chips to find the optimal fixed value
clear all; clc; close all;

addpath('spgl1-2.1');

% Test variants and files
test_classes = {'2S1', 'BMP2_SN_9563', 'T72_SN_132'};
gaussWidths = [1.0, 2.0, 3.0];

% Select the first chip of each
idxChips = 1;
snr = 20;

% Define constants matching raw preprocessing
velLight = 299792458;
numRangeBins = 100;
numFreqBins = 100;
bandwidth = 521e6;
fLower = 9.6e9 - bandwidth/2;
f = linspace(fLower, fLower + bandwidth, numRangeBins).';
thetas = (-1.5:0.03:1.5-0.03);

% Azimuth Basis Centers
AzimuthBasisCenterSpacing = 0.4; % Degrees
numAzimuthBasisCenters = round(3/AzimuthBasisCenterSpacing);
azimuthBasisCenters = 90 + linspace(-1.5, 1.5, numAzimuthBasisCenters);

% Spatial grid points
L = 30;
xGrids = -L/2:0.3:L/2-0.3;
yGrids = -L/2:0.3:L/2-0.3;

[X, Y] = meshgrid(xGrids, yGrids);
Xp = repmat(X(:)', numFreqBins, 1);
Yp = repmat(Y(:)', numFreqBins, 1);

bisectorAngles = 90 + thetas;
azimuthVals = bisectorAngles;
pulseSelectIDx = 1:100;

dist = abs(azimuthVals.' - azimuthBasisCenters);

for idxClass = 1:length(test_classes)
    className = test_classes{idxClass};
    fprintf('\n=========================================\n');
    fprintf('Testing Class %s, Chip 1\n', className);
    fprintf('=========================================\n');
    
    % Load PH
    PH = load(sprintf('../data/phase_histories/%s_PH.mat', className));
    
    % Prepare operator components
    depression = PH.depression(idxChips);
    
    % Target signal
    y1 = PH.arr_img_fft_polar(:, :, idxChips);
    y1 = y1(:);
    sigma_n = sqrt(2)*norm(y1)*10^(-snr/20);
    
    % Construct dictionary matrix template A_mod
    numAzimuthBinsTotal = length(azimuthVals);
    A_mod = zeros(numFreqBins, numRangeBins^2, numAzimuthBinsTotal);
    for idxPulses = 1:numAzimuthBinsTotal
        A_mod(:, :, idxPulses) = 1/sqrt(numFreqBins)*exp(1i*4*pi*f*...
            cosd(depression)/velLight.*(Xp.*cosd(azimuthVals(idxPulses)) +...
            Yp.*sind(azimuthVals(idxPulses))));
    end
    
    % Reconstruct with different gaussWidth values
    for idxG = 1:length(gaussWidths)
        gw = gaussWidths(idxG);
        
        % Slice A_mod outside to optimize speed
        A_mod_sliced = A_mod(:, :, pulseSelectIDx);
        
        % We calculate reconstruction error
        groups_l12 = [1:80000]; % standard L1
        
        tic;
        ee = reconError(y1, gw, pulseSelectIDx, dist, azimuthVals, numFreqBins, numRangeBins, A_mod_sliced, groups_l12, sigma_n);
        t_elap = toc;
        
        fprintf('  gaussWidth = %.1f: Reconstruction Error = %.4e (Time = %.2f s)\n', gw, ee, t_elap);
    end
end
fprintf('\nGaussWidth test complete!\n');
