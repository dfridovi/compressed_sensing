% Parameters 
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
BLOCK_SIZE = 40;
ALPHA = 1.0;
BASIS_OVERSAMPLING = 1.0;
OVERLAP_PERCENT = 0.5;

img = imread([IMAGE_PATH IMAGE_NAME]);
img = rgb2gray(img);

blocks = getBlocks(img,BLOCK_SIZE, OVERLAP_PERCENT);
img_reassembled = assembleBlocks(blocks, BLOCK_SIZE,size(img), OVERLAP_PERCENT);

imshow(img_reassembled)