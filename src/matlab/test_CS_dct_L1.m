%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run L1 compressed sensing on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [500, 500];
BLOCK_SIZE = 20;
ALPHA = [1.0];

OVERLAP_PERCENT = 0.5;
BASIS_OVERSAMPLING = 0.4;

% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

for i = 1:length(ALPHA)
   alpha = ALPHA(i);
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
   
   [M, N, B] = size(blocks);
   [dct_basis, block_coefficients] = compressedSenseDCTL1(blocks, alpha, ...
                                                          BASIS_OVERSAMPLING);
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);
                               
    % Save coefficients to file.
    filename = sprintf('../../reconstructions/matlab figures/cs_dct_size%dx%d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.mat', ...
        IMAGE_SIZE(1), IMAGE_SIZE(2), floor(alpha), 10*mod(alpha, 1), ...
        floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
        floor(BASIS_OVERSAMPLING), 10*mod(BASIS_OVERSAMPLING, 1));
    save(filename, 'block_coefficients');
                               
                               
    % Display.
  	figure;
    imshow(reconstruction, []);
    %title(sprintf('Alpha: %f', alpha));
    
    filename = sprintf('../../reconstructions/matlab figures/cs_dct_size%dx%d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.png', ...
        IMAGE_SIZE(1), IMAGE_SIZE(2), floor(alpha), 10*mod(alpha, 1), ...
        floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
        floor(BASIS_OVERSAMPLING), 10*mod(BASIS_OVERSAMPLING, 1));
    saveas(gcf, filename);
end
