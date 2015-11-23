%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Blockwise sparsity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
RATIO = [0.75, 0.5, 0.1];
OVERLAP_PERCENT = 0;
GAMMA = [0.015];

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));



k = BLOCK_SIZE * BLOCK_SIZE;
blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);

[M, N, B] = size(blocks);
[dct_basis, all_coefficients] = compressDCTL0(blocks, k);
block_coefficients = zeros(size(all_coefficients));
block_nonzero = zeros(1,B);
for j = 1:B
    I = find(abs(all_coefficients(:,j)) >= GAMMA*abs(max(all_coefficients(:,j))));
    block_coefficients(I,j) = all_coefficients(I,j);
    block_nonzero(j) = numel(I);
end
reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
    M, N);
reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
    IMAGE_SIZE, OVERLAP_PERCENT);

% Display.
%     figure;
%     imshow(reconstruction, []);
%     title(sprintf('Compression ratio: %2.1f%%, Gamma: %1.3f%%',...
%         100 * sum(block_nonzero)/numel(img), GAMMA(i)));

error = sqrt(sum(sum((img-reconstruction).^2)));
sparsity = 100 * sum(block_nonzero)/numel(img);
block_nonzero = reshape(block_nonzero, IMAGE_SIZE(1)/BLOCK_SIZE,...
    IMAGE_SIZE(2)/BLOCK_SIZE);


    
%%
figure;

subplot(1,2,1); % original image
imshow(img, []);
title('Original Image');

subplot(1,2,2);
imshow(block_nonzero', []);  
originalSize2 = get(gca, 'Position');
h = colorbar;
ylabel(h, 'k (number of non-zero elements)');
set(gca, 'Position', originalSize2);
title('Blockwise Sparsity');