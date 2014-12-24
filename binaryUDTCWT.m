function histB = binaryUDTCWT(img,wlevels,startlevel,numfeatures,includeavg,mask)
% histB = binaryUDTCWT(img,wlevels,startlevel,mask)
%           Find histogram of binary phase of undecimated DT-CWT (UDT-CWT)
%       inputs:
%           img - 2D matrix of grayscale image
%               - cell array of wavelet coefficients (result of UDT-CWT)
%           wlevels - Total decomposition level (default = log2(min(size(img)))-3);
%           startlevel - Finest included level  (default = 2);
%           numfeatures - number of features produced by histogram
%                         (default = 2^(2*(wlevels-startlevel+1)))
%           includeavg - also include mean of all subbands (default = 0)
%           mask - 2D matrix defines region of interest
%       output:
%           histB - concatenated histogram of bit-planes of 6 subbands
%                   a number of features = 6*ceil(numfeatures/6);
%
%   v1.0 14-02-14 Pui Anantrasirichai, University of Bristol
%   please cite "Robust texture features for blurred images using undecimated dual-tree complex wavelets", 
%   N. Anantrasirichai, J. Burn and David Bull. In Proceedings of the IEEE International Conference on Image Processing (ICIP 2014)


% check inputs
% ------------
if size(img,3)>1
    img = rgb2gray(img);
end
if (nargin < 2) || isempty(wlevels)
    if ~iscell(img)
        wlevels = log2(min(size(img)))-3;
        disp(['Total decomposition level is not defined. L = ',num2str(wlevels),' is used.']);
    else
        wlevels = length(img) - 1;
    end
end
if (nargin < 3) || isempty(startlevel)
    startlevel = 2;
    disp('Finest included level is not defined. l_0 = 2 is used.');
end
if (nargin < 4) || isempty(numfeatures) || (numfeatures==0)
    rangeh = 0:(2^(2*(wlevels-startlevel+1)) - 1);
else
    eachsub = ceil(numfeatures/6);
    steph  = (2^(2*(wlevels-startlevel+1))-1)/(eachsub-1);
    rangeh = 0:steph:(2^(2*(wlevels-startlevel+1)) - 1);
end
if (nargin < 5) || isempty(includeavg)
    includeavg = 0;
end

% UDTCWT transformation
% ---------------------
if ~iscell(img)
    addpath('./UDTCWT/');
    [Faf, ~] = NDAntonB2; %(Must use ND filters for both)
    [af, ~] = NDdualfilt1;
    % find maximal decomposition level
    maxlevel = min(wlevels, floor(log2(min(size(img))/5)));
    % wavelet transform
    w = NDxWav2DMEX(double(img), maxlevel, Faf, af, 1);
    % if more levels are required
    if wlevels > maxlevel
        lowcoef = (w{maxlevel+1}{1}{1}+w{maxlevel+1}{1}{2}+w{maxlevel+1}{2}{1}+w{maxlevel+1}{2}{2});
        morelevels = wlevels - maxlevel;
        wmore = NDxWav2DMEX(double(lowcoef), morelevels, af, af, 1);
        for level = 1:morelevels+1
            w{maxlevel+level} = wmore{level};
        end
    end
    % image dimenstion
    [height, width] = size(img);
else
    w = img;
    % image dimenstion
    [height, width] = size(w{1}{1}{1}{1});
end

% check if mask is defined
% ------------------------
if (nargin < 6) || isempty(mask)
    mask = ones(height, width);
end

% Create bit-planes for each subband and generate histograms
% ----------------------------------------------------------
histindlength = length(rangeh);
histB = zeros(1,6*histindlength + includeavg*histindlength);
for c = 1:2
    for d = 1:3
        bitplanes = zeros(height, width);
        for level = startlevel:wlevels
            realcoef = w{level}{1}{c}{d};
            imgcoef  = w{level}{2}{c}{d};
            bitplanes = bitplanes + (realcoef>0)*(2^(2*(level-startlevel)));
            bitplanes = bitplanes + (imgcoef >0)*(2^(2*(level-startlevel)+1));
        end
        % histogram
        if (nargin >= 6)
            histQ1 = hist(bitplanes(mask(:)>0),rangeh);
        else
            histQ1 = hist(bitplanes(:),rangeh);
        end
        histQ1 = histQ1/sum(histQ1);
        histB(((c-1)*3+d-1)*histindlength + (1:histindlength)) = histQ1;
    end
end
% add histogram of mean of all subbands
if includeavg
    bitplanes = zeros(height, width);
    for level = startlevel:wlevels
        realcoef = zeros(height, width);
        imgcoef = zeros(height, width);
        for c = 1:2
            for d = 1:3
                realcoef = realcoef + w{level}{1}{c}{d};
                imgcoef  = imgcoef + w{level}{2}{c}{d};
            end
        end
        realcoef = realcoef/6;
        imgcoef  = imgcoef/6;
        bitplanes = bitplanes + (realcoef>0)*(2^(2*(level-startlevel)));
        bitplanes = bitplanes + (imgcoef >0)*(2^(2*(level-startlevel)+1));
    end
    % histogram
    if (nargin >= 6)
        histQ1 = hist(bitplanes(mask(:)>0),rangeh);
    else
        histQ1 = hist(bitplanes(:),rangeh);
    end
    histQ1 = histQ1/sum(histQ1);
    histB(6*histindlength + (1:histindlength)) = histQ1;
end