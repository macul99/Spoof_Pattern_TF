function bsifdemo_using_prelearnt_filters()

filename=['./texturefilters/ICAtextureFilters_11x11_8bit'];
load(filename, 'ICAtextureFilters');

img=double(rgb2gray(imread('UniversityOfOulu.jpg')));

% Three ways of using bsif.m :

% the image with grayvalues replaced by the BSIF bitstrings
bsifcodeim= bsif(img,ICAtextureFilters,'im');

% unnormalized BSIF code word histogram
bsifhist=bsif(img,ICAtextureFilters,'h');

% normalized BSIF code word histogram
bsifhistnorm=bsif(img, ICAtextureFilters,'nh');


% visualization
figure;
subplot(1,2,1); imagesc(img);colormap('gray');axis image;colorbar
subplot(1,2,2); imagesc(bsifcodeim); axis image;colorbar

figure;
subplot(1,2,1); bar(bsifhist);
subplot(1,2,2); bar(bsifhistnorm);



