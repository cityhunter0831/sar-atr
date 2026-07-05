clc;
clear all;
close all;

addpath('mtimesx')
path2PH=('../data/phase_histories/');
path2coeff=('../data/recovered_coefficients/');
sample_start=1; %Change sample_start to start generating from that sample
sample_end=1; %This is mostly dummy variable unless line 36 is commented


fileNamePrefix = {'2S1','BMP2_SN_9563','BMP2_SN_9566','BMP2_SN_C21',...
                'BTR70_SN_C71','T72_SN_132','T72_SN_812','T72_SN_S7','ZSU_23_4'};
shiftsy =[0,0.15,0, 0.15];
shiftsx =[0,0, 0.15,0.15];
array_dtheta=[-6:0.25:-0.25 0.25:0.25:6];
tol_add=[-ones(1,9)*0.01 -0.03 0 -0.01 -0.03 0 0 -0.02 zeros(1,3) 0.02 ...
    zeros(1,2) 0.02*ones(1,4) zeros(1,2) 0.02 zeros(1,6) -0.02 ...
    zeros(1,2) -0.02 0 -0.02 -0.03 zeros(1,2) -0.02 0 -0.02 -0.03];
tol_add=flip(-tol_add);

for idxClass = 1:length(fileNamePrefix)

    PH=load(sprintf('%s%s_PH',path2PH,fileNamePrefix{idxClass}));
    RC=load(sprintf('%s%s',path2coeff,fileNamePrefix{idxClass}));
    class_folder=sprintf('../data/gen_aug_data/%s',fileNamePrefix{idxClass});
    if ~isfolder(class_folder)
       mkdir(class_folder)
    end
    numRangeBins = 100;

    
    numTrainingSamples = size(RC.y_recovered,3);
    warning('Next Line will over-ride sample_end to iterate over all remaining samples');
    sample_end=numTrainingSamples;
    
    taylorWindow = kron(taylorwin(100,4,-35),taylorwin(100,4,-35).');
    %thetas = [-5.5:0.03:-1.5-0.03 linspace(-1.5,1.5,100) 1.53:0.03:5.5 ] ;
    thetas = [-7.5:0.03:-1.5-0.03 linspace(-1.5,1.5,100) 1.53:0.03:7.5 ] ;

    bisectorAngles = 90 + thetas;
    azimuthVals = bisectorAngles;
    
    
    
    numFreqBins = 100;
    f_center = 9.6e9;
    bandwidth = 521e6;
    pixelResolutionMSTAR = 0.202;
    delF = bandwidth/100;
    fLower = f_center - bandwidth/2;
    f = linspace(fLower,fLower + bandwidth,100 ).';
    
    
    AzimuthBasisCenterSpacing = 0.4; % Degrees
    numAzimuthBasisCenters = round(3/AzimuthBasisCenterSpacing);
  
    
    
    
    %THe angular and spatial grid points
    
    L=30;
    azimuthBasisCenters = 90+ linspace(-1.5,1.5,numAzimuthBasisCenters);
    
    
    bisectorRep = repmat(bisectorAngles,length(f),1);
    
    
    % Using range resolution=0.3
    xGrids = -L/2:0.3:L/2-0.3;
    yGrids = -L/2:0.3:L/2-0.3;
    
    [X,Y] = meshgrid(xGrids,yGrids);
    Xp = repmat(X(:)',numFreqBins,1);
    Yp = repmat(Y(:)',numFreqBins,1);
    X = X(:);
    Y = Y(:);
    F = repmat(f,1,numRangeBins^2);
    
    thetas1 = linspace(-1.5,1.5,100);
    fRep  = repmat(f,1,numFreqBins);
    
    velLight = 299792458;
    rangeResolution =  velLight/(2*bandwidth);
    [X_rot, Y_rot] = meshgrid(-64:63, -64:63);
    BasisFuncTemp=[];
    bb = bisectorAngles(((bisectorAngles >=90-1.5 ) & (bisectorAngles <= 90+1.5)));
    
    numAzimuthBinsTotal = length(azimuthVals);
    pulseSelectIDx = 1:length(azimuthVals);
    

    for idxTrain = sample_start:sample_end
        gaussWidth = RC.gaussWidthStore(idxTrain);
        dist = abs(bb.' - azimuthBasisCenters);
        BasisFuncTemp = exp(-0.5*dist.^2/gaussWidth^2);
        normBasisFunc = diag((sum(BasisFuncTemp.^2,1)).^0.5);
        
        dist = abs(azimuthVals.' - azimuthBasisCenters);
        BasisFunc = exp(-0.5*dist.^2/gaussWidth^2);
        BasisFunc = BasisFunc/normBasisFunc;
        
        selected_idx = RC.selected_indices(idxTrain);
        depression = PH.depression(selected_idx);
        
        % Memory-efficient loop-based evaluation of y_recovered_unshifted (reduces memory from 4GB to 16MB)
        xx = reshape(RC.x_recovered(:,idxTrain), numRangeBins^2, size(BasisFunc,2));
        xx_interp = xx * BasisFunc.';
        
        y_recovered_unshifted = zeros(numFreqBins, length(azimuthVals));
        const_val = 1i * 4 * pi * f / velLight * cosd(depression);
        for p = 1:length(azimuthVals)
            dist_xyz = X * cosd(azimuthVals(p)) + Y * sind(azimuthVals(p));
            phase_term = exp(const_val * dist_xyz.');
            y_recovered_unshifted(:, p) = 1/sqrt(numFreqBins) * (phase_term * xx_interp(:, p));
        end

        %Using this discrete approach, we could possibly save full 128X128 
        %images
        imgTrain =zeros((length(array_dtheta)+1)*length(shiftsy),64,64);
        aziTrain=zeros((length(array_dtheta)+1)*length(shiftsy),1);
        elev=zeros((length(array_dtheta)+1)*length(shiftsy),1);
        countIm=1;
        
        for idxShifts =1:length(shiftsy)
            fprintf('processing class %d, image %d, shift %d \n'...
                ,idxClass,idxTrain,idxShifts);
            fNew = linspace(fLower*sind(90 + 1.5),fLower + bandwidth,100 ).';
            yy =   (4*pi/velLight*(fNew)*cosd(depression));% (4*pi/velLight*f*cosd(depression)); %
            xx1 = thetas1.';
            xx = (4*pi/velLight*f(end)*cosd(depression)*cosd(90+xx1));
            [XX,YY] = meshgrid(xx,yy);
            pointsOrig = [XX(:).';YY(:).'];
            
            XX = reshape(XX,100,100);
            YY = reshape(YY,100,100);
            
            thetaRep = repmat(thetas1,100,1);
            
            k_1 =  (4*pi/velLight*cosd(depression)*(fRep).*cosd(90+thetaRep));
            k_2 =  (4*pi/velLight*cosd(depression)*(fRep).*sind(90+thetaRep));
            
            % Compute fast phase shift instead of re-evaluating the 4GB A_mod dictionary
            fRep1 = repmat(f,1,length(azimuthVals));
            phase_shift = exp(1i*4*pi*f/velLight*cosd(depression) * (shiftsx(idxShifts)*cosd(azimuthVals) + shiftsy(idxShifts)*sind(azimuthVals)));
            y_recovered1 = y_recovered_unshifted .* phase_shift;
            
            % 0 degree case
            %y =  (y_recovered1(:,find((bisectorAngles >=90-1.5 ) & (bisectorAngles <= 90+1.5)))) ;
            y =  (y_recovered1(:,((bisectorAngles >=90-1.5) & (bisectorAngles <= 90+1.5)))) ;

            yTotal = y;
            ss = scatteredInterpolant(k_1(:),k_2(:),yTotal(:),'natural','nearest');
            arr_img_fft_cart1 = reshape(ss(XX(:),YY(:)),100,100);
            arr_img_fft_cart = zeros(128,128);
            arr_img_fft_cart(14:113,14:113) =   taylorWindow.*arr_img_fft_cart1 ;
            arr_img_recons = ifftshift(ifft2(arr_img_fft_cart));
            
            
            %yTotal= exp(1i*4*pi*fRep*cosd(depression)/velLight.*...
            %    ( shiftsy(idxShifts).*sind(90+thetaRep) )).*PH.arr_img_fft_polar(:,:,selected_idx);
            yTotal= exp(1i*4*pi*fRep*cosd(depression)/velLight.*...
                ( shiftsy(idxShifts).*sind(90+thetaRep) +shiftsx(idxShifts).*cosd(90+thetaRep) ))...
                .*PH.arr_img_fft_polar(:,:,selected_idx);
            normT= norm(yTotal,'fro');
            ss = scatteredInterpolant(k_1(:),k_2(:),yTotal(:),'natural','nearest');
            arr_img_fft_cart1 = reshape(ss(XX(:),YY(:)),100,100);
            arr_img_fft_cart = zeros(128,128);
            arr_img_fft_cart(14:113,14:113) =   taylorWindow.*arr_img_fft_cart1 ;
            arr_img_original = ifftshift(ifft2(arr_img_fft_cart));
            arr_img_residual = arr_img_original - arr_img_recons;
            imgTrain(countIm,:,:) = arr_img_original(32:95,32:95);
            aziTrain(countIm) = PH.arr_azi(selected_idx);
            elev(countIm) = depression;
            countIm = countIm+1;
            
            for idxtheta=1:length(array_dtheta)
                dtheta=array_dtheta(idxtheta);
                
                %y = (y_recovered1(:,((bisectorAngles >=90-1.5+dtheta) & ...
                %    (bisectorAngles <= 90+1.5+(dtheta+tol_add(1,idxtheta))))));
                y = (y_recovered1(:,((bisectorAngles >=90-1.5-dtheta) & ...
                    (bisectorAngles <= 90+1.5-(dtheta+tol_add(1,idxtheta))))));
                yTotal = normT*y/norm(y,'fro') ;
                ss = scatteredInterpolant(k_1(:),k_2(:),yTotal(:),'natural','nearest');
                arr_img_fft_cart1 = reshape(ss(XX(:),YY(:)),100,100);
                arr_img_fft_cart = zeros(128,128);
                arr_img_fft_cart(14:113,14:113) =   taylorWindow.*arr_img_fft_cart1 ;
                arr_img_recons = ifftshift(ifft2(arr_img_fft_cart));
                X_rot_new = X_rot * cosd(dtheta) + Y_rot * sind(dtheta);
                Y_rot_new = -X_rot * sind(dtheta) + Y_rot * cosd(dtheta);
                img_residual_rot = interp2(-64:63, -64:63, arr_img_residual, X_rot_new, Y_rot_new, 'linear', 0);
                arr_img_recons = arr_img_recons + img_residual_rot;

                imgTrain(countIm,:,:) = arr_img_recons(32:95,32:95);

                aziTrain(countIm) = PH.arr_azi(selected_idx)+dtheta;
                elev(countIm) = depression;
                countIm = countIm+1;
            end
        end
        %Only save once at the end of all subpixel shifts of the sample
        save(sprintf('%s\\sample_%d',class_folder,idxTrain),'imgTrain','aziTrain','elev');
    end
end