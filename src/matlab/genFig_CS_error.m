%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Error vs. Gamma ( Compressed Sensing DCT Lasso )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';
IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
OVERLAP_PERCENT = 0;

% Import original the image.
img = double(imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])),...
    IMAGE_SIZE));


path = '../../reconstructions/matlab figures/cs_dct_lasso/';

ALPHA = [0.01 0.1 1 10];
OS = 0.1:0.1:1.5;
GAMMA = 0.015;

error = zeros(numel(ALPHA), numel(OS));
sparsity = zeros(numel(ALPHA), numel(OS));
block_nonzero = zeros(numel(ALPHA), numel(OS), round(IMAGE_SIZE(1)/BLOCK_SIZE)^2);

for i = 1:numel(ALPHA)
    % Get filename
    alpha = ALPHA(i);
    d1 = floor(alpha);
    d2 = num2str(floor(mod(alpha,1)*10));
    if alpha == 0.01
        d2 = '01';
    end
    for j = 1:numel(OS)
        os = OS(j);
        d3 = floor(os);
        d4 = floor(mod(os,1)*10);
        filename = sprintf('cs_dct_size512x512_alpha%dp%s_overlap0p0_oversample%dp%d',...
            d1, d2, d3, d4);
        
        % Import coefficients and png
        load([path filename '.mat']);
        %reconstruction = double(imread([path filename '.png']));
        M = 8;
        N = 8;
        dct_basis = computeDCTBasis(M, N);
        reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
            M, N);
        reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                                       IMAGE_SIZE, OVERLAP_PERCENT);
        
        error(i,j) = sqrt(sum(sum((img-reconstruction).^2)));
        B = size(block_coefficients,2);

        for n = 1:B
            block_nonzero(i,j,n) = block_nonzero(i,j,n) + numel(find(abs(block_coefficients(:,n)) > ...
                max(max(abs(block_coefficients(:,n))))*GAMMA));
        end
        sparsity(i,j) = 100 * sum(block_nonzero(i,j,:))/numel(img); % percent of coefficients greater than gamma
        
    end
end

    
%%
figure;
epsilon = 1800;
lw = 1;


subplot(2,1,1);
plot(OS, error, 'linewidth', lw)
legend('\alpha = 0.01', '\alpha = 0.1', '\alpha = 1', '\alpha = 10')
xlim([0 1.5]);
xaxis = xlim;
x = [xaxis(1):.01:xaxis(2)];
hold on; plot(x, epsilon*ones(size(x)), '--', 'linewidth', lw);
ylim([0 10^4])


subplot(2,1,2);
plot(sparsity(1,:),error(1,:), sparsity(2,:),error(2,:),...
    sparsity(3,:),error(3,:),sparsity(4,:),error(4,:),'linewidth', lw)

xaxis = xlim;
x = [xaxis(1):.01:xaxis(2)];
hold on; plot(x, epsilon*ones(size(x)), '--', 'linewidth', lw);
ylim([0 10^4])

%%

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
xlabel(sprintf('Percent of coefficeints \n used in reconstruction')); ylabel('Error');

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