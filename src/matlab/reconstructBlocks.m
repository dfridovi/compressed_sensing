function reconstructed_blocks = ...
    reconstructBlocks(basis, block_coefficients, M, N)
    % Reconstruct a set of blocks from the basis and coefficients.
    
    [L, B] = size(block_coefficients);
    reconstructed_blocks = zeros(M, N, B);
    
    for i = 1:B
       coefficients = block_coefficients(:, i);
       reconstruction = basis * coefficients;
       reconstructed_blocks(:, :, i) = reshape(reconstruction, M, N);
    end
end