function reconstructed_blocks = ...
    reconstructBlocks(basis, block_coefficients, M)
    % Reconstruct a set of blocks from the basis and coefficients.
    
    [L, B] = size(block_coefficients);
    reconstructed_blocks = zeros(M, B);
    
    for i = 1:B
       coefficients = block_coefficients(:, i);
       reconstruction = basis * coefficients;
       reconstructed_blocks(:, i) = reconstruction;
    end
end