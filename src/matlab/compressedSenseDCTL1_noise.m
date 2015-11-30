function [dct_basis, block_coefficients, noisy_blocks] = ...
    compressedSenseDCTL1_noise(blocks, alpha, basis_oversampling, sd)
% Sketch the image blocks in the DCT domain. Procedure:
%     1. Choose a random matrix to mix the DCT components.
%     2. Solve the L1-penalized least-squares problem.
% 
%     min_x ||AFx - m||_2^2 + alpha * ||x||_1
%     
%     where: m = sampled image,
%            x = representation,
%            A = mixing matrix,
%            F = DCT basis

    [M, N, B] = size(blocks); % blocks is a 3D array
    noisy_blocks = zeros(size(blocks));
    
    % Generate a mixing matrix.
    mixing = randn(round(M * N * basis_oversampling), M * N);
   
    % Generate the DCT basis.
    dct_basis = computeDCTBasis(M, N);
    
    % Pre-mulitply the dct basis by the mixing matrix.
    basis_premultiplied = mixing * dct_basis;
    
    % Iterate over all blocks. Store coefficients in a 2D array.
    block_coefficients = zeros(M * N, B);
    parfor i = 1:B
       %fprintf('Working on block %d of %d...\n', i, B);
       block = blocks(:, :, i);
       [coefficients, block_measured] = blockCompressedSenseL1_noise(block, alpha, ...
                                             basis_premultiplied, ...
                                             mixing, sd);
       block_coefficients(:, i) = coefficients;
       noisy_blocks(:,:,i) = reshape(inv(mixing)*block_measured, M, N);
    end
end