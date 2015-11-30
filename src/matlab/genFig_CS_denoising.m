%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Denoising with CS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;

OVERLAP_PERCENT = 0;
ALPHA = 1;
OS = 1;

NOISE_SD = 20.0;

% Import original the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

% Add noise.
noisy = img + NOISE_SD * BLOCK_SIZE^2 * randn(size(img));

% Compress.
blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);

[M, N, B] = size(blocks);
[dct_basis, block_coefficients, noisy_blocks] = compressedSenseDCTL1_noise(blocks, ALPHA, OS, NOISE_SD);
reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, M, N);
reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                IMAGE_SIZE, OVERLAP_PERCENT);
noisy_img = assembleBlocks(noisy_blocks,BLOCK_SIZE, ...
                                IMAGE_SIZE, OVERLAP_PERCENT);


% Display.
figure; 
subplot(1, 2, 1); imshow(noisy_img, []); title('Noisy');
subplot(1, 2, 2); imshow(reconstruction, []); title('Reconstructed');