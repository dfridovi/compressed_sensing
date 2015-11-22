% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [50, 50];
BLOCK_SIZE = 8;
RATIO = [0.75, 0.5, 0.1];
OVERLAP_PERCENT = 0;
GAMMA = [.000 .01 .02];

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

imgDCT = blockDCTL0(img,IMAGE_SIZE(1)*IMAGE_SIZE(2)*.5);
basis = computeDCTBasis(IMAGE_SIZE(1), IMAGE_SIZE(2));
imgRE = basis*imgDCT;
imgRE = reshape(imgRE, IMAGE_SIZE);

figure; subplot(1,2,1); imshow(img, []); title('Original')
subplot(1,2,2); imshow(imgRE,[]); title('Reconstruction')