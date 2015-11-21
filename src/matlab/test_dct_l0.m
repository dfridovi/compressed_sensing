%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run Fourier L0 compression on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [500, 500];
BLOCK_SIZE = 2000;
RATIO = [0.2, 0.05, 0.01];
OVERLAP = 0.5;

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE));

figure;
subplot(2, 2, 1);
imshow(img, []);
title('Original Image');

for i = 1:3
   k = RATIO(i) * BLOCK_SIZE;
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP);
   [M, B] = size(blocks);
   [dct_basis, block_coefficients] = compressDCTL0(blocks, k);
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, M);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, IMAGE_SIZE, OVERLAP);

    % Display.
    subplot(2, 2, i+1);
    imshow(reconstruction, []);
    title(sprintf('Compression ratio: %2.1f%%', 100 * RATIO(i)));
end
