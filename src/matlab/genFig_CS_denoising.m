%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Denoising with CS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;

OVERLAP_PERCENT = 0;
ALPHA = 0.1;
OS = 0.5;

NOISE_SD = 10.0;

% Import original the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

% Add noise.
noisy = img + NOISE_SD * randn(size(img));

% Compress.
blocks = getBlocks(noisy, BLOCK_SIZE, OVERLAP_PERCENT);

[M, N, B] = size(blocks);
[dct_basis, block_coefficients] = compressedSenseDCTL1(blocks, ALPHA, OS);
reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, M, N);
reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                IMAGE_SIZE, OVERLAP_PERCENT);


% Display.
figure; 
subplot(1, 2, 1); imshow(noisy, []); title('Noisy');
subplot(1, 2, 2); imshow(reconstruction, []); title('Reconstructed');