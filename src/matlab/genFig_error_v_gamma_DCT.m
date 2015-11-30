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
GAMMA = [0 logspace(-4,-1,19)];

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
%     figure;
%     imshow(reconstruction, []);
%     title(sprintf('Compression ratio: %2.1f%%, Gamma: %1.3f%%',...
%         100 * sum(block_nonzero)/numel(img), GAMMA(i)));
    
    error(i) = sqrt(sum(sum((img-reconstruction).^2)));
    sparsity(i) = 100 * sum(block_nonzero)/numel(img);
end
%%
gamma_plot = [0.015 0.03 0.07];
reconst_plot = zeros(size(img,1),size(img,2),3);

for i = 1:numel(gamma_plot)
   k = BLOCK_SIZE * BLOCK_SIZE;
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
   
   [M, N, B] = size(blocks);
   [dct_basis, all_coefficients] = compressDCTL0(blocks, k);
   block_coefficients = zeros(size(all_coefficients));
   block_nonzero = zeros(1,B);
   for j = 1:B
        I = find(abs(all_coefficients(:,j)) >= gamma_plot(i)*abs(max(all_coefficients(:,j))));
        block_coefficients(I,j) = all_coefficients(I,j);
        block_nonzero(j) = numel(I);
   end
   reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);
   reconst_plot(:,:,i) = reconstruction;
    % Display.
%     figure;
%     imshow(reconstruction, []);
%     title(sprintf('Compression ratio: %2.1f%%, Gamma: %1.3f%%',...
%         100 * sum(block_nonzero)/numel(img), GAMMA(i)));
    
    error_plot(i) = sqrt(sum(sum((img-reconstruction).^2)));
    sparsity_plot(i) = 100 * sum(block_nonzero)/numel(img);
end


    
%%
figure;
epsilon = 1800;
lw = 1;

subplot(2,3,1); plot(GAMMA,error, 'linewidth', lw);
xaxis = xlim;
x = [xaxis(1):.01:xaxis(2)];
hold on; plot(x, epsilon*ones(size(x)), '--', 'linewidth', lw);
plot([0 gamma_plot], [0 error_plot], 'ko');
text(0,0, '  A');
text(gamma_plot(1), error_plot(1), '  B');
text(gamma_plot(2), error_plot(2), '  C');
text(gamma_plot(3), error_plot(3), '  D');
hold off;
xlabel('Gamma'); ylabel('Error');

subplot(2,3,4); plot(sparsity, error, 'linewidth', lw);
xaxis = xlim;
x = [xaxis(1):.01:xaxis(2)];
hold on; plot(x, epsilon*ones(size(x)), '--', 'linewidth', lw);
plot([100 sparsity_plot], [0 error_plot], 'ko');
text(100,0, '  A');
text(sparsity_plot(1), error_plot(1), '  B');
text(sparsity_plot(2), error_plot(2), '  C');
text(sparsity_plot(3), error_plot(3), '  D');
hold off;
xlabel(sprintf('Percent of coefficients \n used in reconstruction')); ylabel('Error');

range = 200:300;

subplot(2,3,2); % original image
imshow(img(range, range), []);
title('(A) Original Image');

subplot(2,3,3); % gamma = 0.015
imshow(reconst_plot(range,range,1), []);
title(sprintf('(B) Error: %4.0f', error_plot(1)));

subplot(2,3,5); % gamma = 0.03
imshow(reconst_plot(range,range,2), []);
title(sprintf('(C) Error: %4.0f', error_plot(2)));

subplot(2,3,6); % gamma = 0.07
imshow(reconst_plot(range,range,3), []);
title(sprintf('(D) Error: %4.0f', error_plot(3)));