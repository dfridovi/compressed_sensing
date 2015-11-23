%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Comparision of domains: Image, Fourier, DCT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
RATIO = [0.75, 0.5, 0.1];
OVERLAP_PERCENT = 0;
GAMMA = fliplr([0 logspace(-1,2,19)]);
GAMMA = [0.24:-.0001:.23];
epsilon = 1800;

% Import the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));

 
    %%
    
    figure;
    x = logspace(-5, 0, 25);
    
    gamma_fourier = 0.0102;
    gamma_DCT = 0.0146;
    gamma_img = 0.236;
    
    %% Image Domain
    
    k = BLOCK_SIZE * BLOCK_SIZE;
    blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
    [M, N, B] = size(blocks);
    block_coefficients = zeros(M*N, B);
    block_nonzero = zeros(1,B);
    for j = 1:B
        all_coefficients = reshape(blocks(:,:,j), M*N, 1); % for image domain only
        I = find(abs(all_coefficients(:)) >= gamma_img*abs(max(all_coefficients(:))));
        block_coefficients(I,j) = all_coefficients(I);
        block_nonzero(j) = numel(I);
        block_reshaped(:,:,j) = reshape(block_coefficients(:,j), M, N);
    end
    
    reconstructed_blocks = block_reshaped;
    reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
        IMAGE_SIZE, OVERLAP_PERCENT);
    
    subplot(2,3,1);
    imshow(reconstruction, []);
    title(sprintf('Image Domain \n %2.1f%% of Coeffiecients', ...
        100*(sum(block_nonzero)/numel(img))));
    subplot(2,3,4);
    coeff_vector = reshape(blocks, numel(blocks), 1);
    hist(abs(coeff_vector)/abs(max(coeff_vector)), x);
         sh=findall(gcf,'marker','*');
         delete(sh);
    title('Image Domain');
    set(gca,'xscale','log')
     xlim([10^-5 10^0]);
     
     ymax = ylim;
     y = 0:ymax(2);
     hold on;
     h = plot(gamma_img*ones(size(y)), y, 'linewidth', 2);
     legend(h, 'gamma', 'Location', 'NorthWest');
     
     ylabel('Count')
     xlabel('Normalized Coefficients')
    
    %% Fourier
    k = BLOCK_SIZE * BLOCK_SIZE;
    blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
    
    [M, N, B] = size(blocks);
    [dct_basis, all_coefficients] = compressFourierL0(blocks, k);
    block_coefficients = zeros(size(all_coefficients));
    block_nonzero = zeros(1,B);
    for j = 1:B
        I = find(abs(all_coefficients(:,j)) >= gamma_fourier*abs(max(all_coefficients(:,j))));
        block_coefficients(I,j) = all_coefficients(I,j);
        block_nonzero(j) = numel(I);
    end
    reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
        M, N);
    reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
        IMAGE_SIZE, OVERLAP_PERCENT);
    
    subplot(2,3,2);
    imshow(reconstruction, []);
    title(sprintf('Fourier Domain \n %2.1f%% of Coeffiecients', ...
        100*(sum(block_nonzero)/numel(img))));
    subplot(2,3,5);
    coeff_vector = reshape(all_coefficients, numel(all_coefficients), 1);
    hist(abs(coeff_vector)/abs(max(coeff_vector)), x);
         sh=findall(gcf,'marker','*');
         delete(sh);
    title('Fourier Domain');
    set(gca,'xscale','log')
     xlim([10^-5 10^0]);
     
         ymax = ylim;
     y = 0:ymax(2);
     hold on;
     plot(gamma_fourier*ones(size(y)), y, 'linewidth', 2);
     xlabel('Normalized Coefficients')
    
    %% DCT
    k = BLOCK_SIZE * BLOCK_SIZE;
    blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
    
    [M, N, B] = size(blocks);
    [dct_basis, all_coefficients] = compressDCTL0(blocks, k);
    block_coefficients = zeros(size(all_coefficients));
    block_nonzero = zeros(1,B);
    for j = 1:B
        I = find(abs(all_coefficients(:,j)) >= gamma_DCT*abs(max(all_coefficients(:,j))));
        block_coefficients(I,j) = all_coefficients(I,j);
        block_nonzero(j) = numel(I);
    end
    reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
        M, N);
    reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
        IMAGE_SIZE, OVERLAP_PERCENT);
    
    subplot(2,3,3);
    imshow(reconstruction, []);
    title(sprintf('DCT Domain \n %2.1f%% of Coeffiecients', ...
        100*(sum(block_nonzero)/numel(img))));
    subplot(2,3,6);
    coeff_vector = reshape(all_coefficients, numel(all_coefficients), 1);
    hist(abs(coeff_vector)/abs(max(coeff_vector)), x);
         sh=findall(gcf,'marker','*');
         delete(sh);
    title('DCT Domain');
    set(gca,'xscale','log')
    xlim([10^-5 10^0]);
    
        ymax = ylim;
     y = 0:ymax(2);
     hold on;
     plot(gamma_DCT*ones(size(y)), y, 'linewidth', 2);
     xlabel('Normalized Coefficients')

    
    
    
    
