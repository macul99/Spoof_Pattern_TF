function LPQdesc = lpq(img,winSize,mode)
% https://stackoverflow.com/questions/23548863/converting-a-specific-matlab-script-to-python
%% Defaul parameters
% Local window size
if nargin<2 || isempty(winSize)
    winSize=3; % default window size 3
end

rho=0.90; % Use correlation coefficient rho=0.9 as default

% Local frequency estimation (Frequency points used [alpha,0], [0,alpha], [alpha,alpha], and [alpha,-alpha]) 
if nargin<4 || isempty(freqestim)
    freqestim=1; %use Short-Term Fourier Transform (STFT) with uniform window by default
end
STFTalpha=1/winSize;  % alpha in STFT approaches (for Gaussian derivative alpha=1) 
sigmaS=(winSize-1)/4; % Sigma for STFT Gaussian window (applied if freqestim==2)
sigmaA=8/(winSize-1); % Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

% Output mode
if nargin<5 || isempty(mode)
    mode='nh'; % return normalized histogram as default
end

% Other
convmode='valid'; % Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates image with zeros).


%% Check inputs
if size(img,3)~=1
    error('Only gray scale image can be used as input');
end
if winSize<3 || rem(winSize,2)~=1
   error('Window size winSize must be odd number and greater than equal to 3');
end
if sum(strcmp(mode,{'nh','h','im'}))==0
    error('mode must be nh, h, or im. See help for details.');
end


%% Initialize
img=double(img); % Convert image to double
r=(winSize-1)/2; % Get radius from window size
x=-r:r; % Form spatial coordinates in window
u=1:r; % Form coordinates of positive half of the Frequency domain (Needed for Gaussian derivative)

%% Form 1-D filters
 % STFT uniform window
    % Basic STFT filters
    w0=(x*0+1);
    w1=exp(complex(0,-2*pi*x*STFTalpha));
    w2=conj(w1);


%% Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
% Run first filter
filterResp=conv2(conv2(img,w0.',convmode),w1,convmode);
% Initilize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency).
freqResp=zeros(size(filterResp,1),size(filterResp,2),8); 
% Store filter outputs
freqResp(:,:,1)=real(filterResp);
freqResp(:,:,2)=imag(filterResp);
% Repeat the procedure for other frequencies
filterResp=conv2(conv2(img,w1.',convmode),w0,convmode);
freqResp(:,:,3)=real(filterResp);
freqResp(:,:,4)=imag(filterResp);
filterResp=conv2(conv2(img,w1.',convmode),w1,convmode);
freqResp(:,:,5)=real(filterResp);
freqResp(:,:,6)=imag(filterResp);
filterResp=conv2(conv2(img,w1.',convmode),w2,convmode);
freqResp(:,:,7)=real(filterResp);
freqResp(:,:,8)=imag(filterResp);

% Read the size of frequency matrix
[freqRow,freqCol,freqNum]=size(freqResp);

%% Perform quantization and compute LPQ codewords
LPQdesc=zeros(freqRow,freqCol); % Initialize LPQ code word image (size depends whether valid or same area is used)
for i=1:freqNum
    LPQdesc=LPQdesc+(double(freqResp(:,:,i))>0)*(2^(i-1));
end

%% Switch format to uint8 if LPQ code image is required as output
if strcmp(mode,'im')
    LPQdesc=uint8(LPQdesc);
end

%% Histogram if needed
if strcmp(mode,'nh') || strcmp(mode,'h')
    LPQdesc=hist(LPQdesc(:),0:255);
end

%% Normalize histogram if needed
if strcmp(mode,'nh')
    LPQdesc=LPQdesc/sum(LPQdesc);
end