% test_blocking.m
% test if blocking and reassembling makes an image 

% Parameters 
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
BLOCK_SIZE = 40;
ALPHA = 1.0;
BASIS_OVERSAMPLING = 1.0;
OVERLAP_PERCENT = .4;


img = imread([IMAGE_PATH IMAGE_NAME]);
img = rgb2gray(img);

%img = ones(size(img));

blocks = getBlocks(img,BLOCK_SIZE, OVERLAP_PERCENT);
img_reassembled = assembleBlocks(blocks, BLOCK_SIZE, size(img), OVERLAP_PERCENT);

imagesc(img_reassembled); colorbar