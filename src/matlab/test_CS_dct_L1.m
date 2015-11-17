%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run Fourier L0 compression on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [500, 500];
BLOCK_SIZE = 20;
ALPHA = [1.0, 0.1, 0.01];
RHO = 0.1;
OVERLAP_PERCENT = 0.5;
BASIS_OVERSAMPLING = 1.0;

% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

figure;
subplot(2, 2, 1);
imshow(img, []);
title('Original Image');

for i = 1:3
   alpha = ALPHA(i);
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
   
   [M, N, B] = size(blocks);
   [dct_basis, block_coefficients] = compressedSenseDCTL1(blocks, RHO, alpha, ...
                                                          BASIS_OVERSAMPLING);
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);

    % Display.
    subplot(2, 2, i+1);
    imshow(reconstruction, []);
    title(sprintf('Compression ratio: %2.1f%%', 100 * RATIO(i)));
end
