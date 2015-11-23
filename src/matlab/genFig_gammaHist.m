%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Gamma Histogram
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


    
%%
figure;
lw = 2;
x = logspace(-5, 0, 25);

subplot(1,2,1); % sparse
num1 = 3613;
hist(abs(all_coefficients(:,num1))/abs(max(all_coefficients(:,num1))), x);
title('Sparse Block');
set(gca,'xscale','log') 
hold on
plot(0.015*ones(size(0:18)), 0:18, 'linewidth', lw);
     sh=findall(gcf,'marker','*');
     delete(sh);
hold off
xlabel('Normalized Coefficient Magnitude');
ylabel('Count');

subplot(1,2,2);
num2 = 2459;
hist(abs(all_coefficients(:,num2))/abs(max(all_coefficients(:,num2))), x);
title('Dense Block')
hold on
h = plot(0.015*ones(size(0:18)), 0:18, 'linewidth', lw);
     sh=findall(gcf,'marker','*');
     delete(sh);
hold off
set(gca,'xscale','log');
xlabel('Normalized Coefficient Magnitude');
ylabel('Count');
legend(h,'Gamma');