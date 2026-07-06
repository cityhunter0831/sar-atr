clc;
clear all;
close all;

addpath('spgl1-2.1');
addpath('mtimesx');

% Load first class data
fileNamePrefix = 'BMP2_SN_9563';
data_PH = load(sprintf('../data/phase_histories/%s_PH', fileNamePrefix));

f_center = 9.6e9;
bandwidth = 521e6;
fLower = f_center - bandwidth/2;
f = linspace(fLower, fLower + bandwidth, 100).';
thetas = (-1.5:0.03:1.5-0.03);
bisectorAngles = 90 + thetas;
azimuthVals = bisectorAngles;
velLight = 299792458;

numFreqBins = length(f);
numRangeBins = numFreqBins;
numAzimuthBinsTotal = length(bisectorAngles);
AzimuthBasisCenterSpacing = 0.4;
numAzimuthBasisCenters = round(3/AzimuthBasisCenterSpacing);

L = 30;
azimuthBasisCenters = 90 + linspace(-1.5, 1.5, numAzimuthBasisCenters);
xGrids = -L/2:0.3:L/2-0.3;
yGrids = -L/2:0.3:L/2-0.3;

[X, Y] = meshgrid(xGrids, yGrids);
Xp = repmat(X(:)', numFreqBins, 1);
Yp = repmat(Y(:)', numFreqBins, 1);
X = X(:);
Y = Y(:);
F = repmat(f, 1, numRangeBins^2);

dist = pdist2(azimuthVals.', azimuthBasisCenters');
BasisFunc = exp(-0.5*dist.^2/1.0^2);
normBasisFunc = diag((sum(BasisFunc.^2, 1)).^0.5);
BasisFunc = BasisFunc/normBasisFunc;
numVariables = size(BasisFunc, 2);
groups_l12 = repmat((1:numRangeBins^2).', 1, numVariables);
groups_l12 = groups_l12(:);

pulseSelectIDx = 1:length(azimuthVals);

% Process 1st chip
idxChips = 1;
depression = data_PH.depression(idxChips);
A_mod = zeros(numFreqBins, numRangeBins^2, numAzimuthBinsTotal);

fprintf('Setting up dictionary matrix A_mod...\n');
tic;
for idxPulses = 1:numAzimuthBinsTotal
    A_mod(:, :, idxPulses) = 1/sqrt(numFreqBins)*exp(1i*4*pi*F*...
        cosd(depression)/velLight.*(Xp.*cosd(azimuthVals(idxPulses)) +...
        Yp.*sind(azimuthVals(idxPulses))));
end
toc;

y1 = data_PH.arr_img_fft_polar(:, :, idxChips);
y1 = y1(:);
snr = 20;
sigma_n = sqrt(2)*norm(y1)*10^(-snr/20);

BasisFunc = exp(-0.5*dist.^2/1.0^2);
normBasisFunc = diag((sum(BasisFunc.^2, 1)).^0.5);
BasisFunc = BasisFunc/normBasisFunc;

A_mod_sliced = A_mod(:, :, pulseSelectIDx);
A = @(x, mode) SAR_operator_gen(x, mode, pulseSelectIDx, numRangeBins, numFreqBins, azimuthVals, BasisFunc, A_mod_sliced);

% Test single run of SPGL1 with iterations=100
fprintf('Running SPGL1 with 100 iterations...\n');
opts = spgSetParms('iscomplex', 1, 'verbosity', 1, 'iterations', 100);
tic;
C = spg_group(A, y1, groups_l12, sigma_n, opts);
t_spg = toc;
fprintf('SPGL1 time: %.2f seconds\n', t_spg);

% Test single run of SPGL1 with iterations=10
fprintf('Running SPGL1 with 10 iterations...\n');
opts = spgSetParms('iscomplex', 1, 'verbosity', 1, 'iterations', 10);
tic;
C = spg_group(A, y1, groups_l12, sigma_n, opts);
t_spg_10 = toc;
fprintf('SPGL1 10 iterations time: %.2f seconds\n', t_spg_10);
