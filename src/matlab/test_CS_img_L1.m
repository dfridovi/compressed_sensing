%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run L1 compressed sensing on an image in the standard basis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
ALPHA = [0.01, 0.1 1.0 10];
OVERLAP_PERCENT = 0;
BASIS_OVERSAMPLING = 0.1:0.1:1.5;

% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

for i = 1:length(ALPHA)
    for j = 1:numel(BASIS_OVERSAMPLING)        
       basis_oversampling = BASIS_OVERSAMPLING(j);
       alpha = ALPHA(i);
       fprintf('ALPHA = %1.1f, OSR = %1.1f\n', alpha, basis_oversampling);
       
       blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);

       [M, N, B] = size(blocks);
       block_coefficients = compressedSenseImgL1(blocks, alpha, ...
                                                              basis_oversampling);
       reconstructed_blocks = reconstructBlocks(eye(M * N), block_coefficients, ...
                                                M, N);
       reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                       IMAGE_SIZE, OVERLAP_PERCENT);

        % Save coefficients to file.
        filename = sprintf('../../reconstructions/matlab figures/cs_img_lasso/cs_img_size%dx%d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.mat', ...
            IMAGE_SIZE(1), IMAGE_SIZE(2), floor(alpha), 10*mod(alpha, 1), ...
            floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
            floor(basis_oversampling), 10*mod(basis_oversampling, 1));
        save(filename, 'block_coefficients');
                               
                               
    % Display.
%   	figure;
%     imshow(reconstruction, []);
    %title(sprintf('Alpha: %f', alpha));
    
    filename = sprintf('../../reconstructions/matlab figures/cs_img_lasso/cs_img_size%dx%d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.png', ...
        IMAGE_SIZE(1), IMAGE_SIZE(2), floor(alpha), 10*mod(alpha, 1), ...
        floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
        floor(basis_oversampling), 10*mod(basis_oversampling, 1));
    scaled_reconstruction = reconstruction / max(max(reconstruction));
    imwrite(scaled_reconstruction, filename);
    end
end
