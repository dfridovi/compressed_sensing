%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run reversed Huber compressed sensing on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [48, 48];
BLOCK_SIZE = 16;
ALPHA = [1.0];
RHO = 0.1;
OVERLAP_PERCENT = 0.5;
BASIS_OVERSAMPLING = 1.0;

% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

for i = 1:length(ALPHA)
   alpha = ALPHA(i);
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
   
   [M, N, B] = size(blocks);
   [dct_basis, block_coefficients] = compressedSenseDCTHuber(blocks, RHO, alpha, ...
                                                             BASIS_OVERSAMPLING);
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);
                               
    % Save coefficients to file.
    filename = sprintf('../../reconstructions/matlab figures/cs_dct_huber_size%dx%d_rho%1dp%1d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.mat', ...
        IMAGE_SIZE(1), IMAGE_SIZE(2), floor(RHO), 10*mod(RHO, 1), floor(alpha), 10*mod(alpha, 1), ...
        floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
        floor(BASIS_OVERSAMPLING), 10*mod(BASIS_OVERSAMPLING, 1));
    save(filename, 'block_coefficients');
                               
                               
    % Display.
    figure;
    imshow(reconstruction, []);
    %title(sprintf('Alpha: %f', alpha));
    
    filename = sprintf('../../reconstructions/matlab figures/cs_dct_huber_size%dx%d_rho%1dp%1d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.png', ...
        IMAGE_SIZE(1), IMAGE_SIZE(2), floor(RHO), 10*mod(RHO, 1), floor(alpha), 10*mod(alpha, 1), ...
        floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
        floor(BASIS_OVERSAMPLING), 10*mod(BASIS_OVERSAMPLING, 1));
    saveas(gcf, filename);
end
