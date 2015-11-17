function [fourier_basis, block_coefficients] = compressFourierL0(blocks, k)
    % Run top 'k' Fourier L0 compression on each block.

    [M, N, B] = size(blocks); % blocks is a 3D array
    block_coefficients = zeros(M * N, B);
    
    for i = 1:B
       block = blocks(:, :, i);
       coefficients = blockFourierL0(block, k);
       block_coefficients(:, i) = coefficients;
    end
    
    fourier_basis = computeFourierBasis(M * N);
end