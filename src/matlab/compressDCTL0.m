function [dct_basis, block_coefficients] = compressDCTL0(blocks, k)
    % Run top 'k' DCT L0 compression on each block.

    [M, B] = size(blocks); % blocks is a 3D array
    block_coefficients = zeros(M, B);
    
    for i = 1:B
       block = blocks(:, i);
       coefficients = blockDCTL0(block, k);
       block_coefficients(:, i) = coefficients;
    end
    
    dct_basis = computeDCTBasis(M);
end