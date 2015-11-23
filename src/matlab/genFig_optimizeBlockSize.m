%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Optimize Block Size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [512, 512];
BLOCK_SIZE = [1 2 4 8 16 32 64 128];
RATIO = [0.75, 0.5, 0.1];
OVERLAP_PERCENT = 0;
GAMMA = fliplr([0 logspace(-4,-1,19)]);
epsilon = 1800;

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

error = zeros(size(BLOCK_SIZE));
sparsity = zeros(size(BLOCK_SIZE));
gamma_star = zeros(size(BLOCK_SIZE));

for i = 1:numel(BLOCK_SIZE)
    
    k = BLOCK_SIZE(i) * BLOCK_SIZE(i);
    blocks = getBlocks(img, BLOCK_SIZE(i), OVERLAP_PERCENT);
    [M, N, B] = size(blocks);
    [dct_basis, all_coefficients] = compressDCTL0(blocks, k);
    block_coefficients = zeros(size(all_coefficients));
    block_nonzero = zeros(1,B);
    
    error_inter = epsilon+1;
    sparsity_inter = 0;
    n = 0;
    % Calculate gamma
    while(error_inter > epsilon)
        n = n+1;
        for j = 1:B
            I = find(abs(all_coefficients(:,j)) >= GAMMA(n)*abs(max(all_coefficients(:,j))));
            block_coefficients(I,j) = all_coefficients(I,j);
            block_nonzero(j) = numel(I);
        end
        reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
            M, N);
        reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE(i), ...
            IMAGE_SIZE, OVERLAP_PERCENT);
        error_prev = error_inter;
        error_inter = sqrt(sum(sum((img-reconstruction).^2)));
        sparsity_prev = sparsity_inter;
        sparsity_inter = sum(block_nonzero);
        
    end
    
    gamma_star(i) = GAMMA(n);
    error(i) = error_prev;
    sparsity(i) = sparsity_prev;
 
    
end

%%
figure;subplot(2,1,1); plot(BLOCK_SIZE, error); ylabel('error');
subplot(2,1,2);
plot(BLOCK_SIZE, sparsity); xlabel('Block size');ylabel('k - number of nonzero elements');