%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Error vs. Gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
RATIO = [0.75, 0.5, 0.1];
OVERLAP_PERCENT = 0;
GAMMA = [0:.005:.1];

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

% figure;
% subplot(2, 2, 1);
% imshow(img, []);
% title('Original Image');
error = zeros(size(GAMMA));
sparsity = zeros(size(GAMMA));

for i = 1:numel(GAMMA)
   k = BLOCK_SIZE * BLOCK_SIZE;
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
   
   [M, N, B] = size(blocks);
   [dct_basis, all_coefficients] = compressDCTL0(blocks, k);
   block_coefficients = zeros(size(all_coefficients));
   block_nonzero = zeros(1,B);
   for j = 1:B
        I = find(abs(all_coefficients(:,j)) >= GAMMA(i)*abs(max(all_coefficients(:,j))));
        block_coefficients(I,j) = all_coefficients(I,j);
        block_nonzero(j) = numel(I);
   end
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);

    % Display.
%     subplot(2, 2, i+1);
%     imshow(reconstruction, []);
%     title(sprintf('Compression ratio: %2.1f%%',...
%         100 * sum(block_nonzero)/numel(img)));
    
    error(i) = sqrt(sum(sum((img-reconstruction).^2)));
    sparsity(i) = 100 * sum(block_nonzero)/numel(img);
end
%%
figure; subplot(2,1,1); plot(GAMMA,error, '-o');xlabel('Gamma'); ylabel('error');
subplot(2,1,2); plot(sparsity, error,'-o'); xlabel('Compression Ratio');