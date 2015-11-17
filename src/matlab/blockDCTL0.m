function coefficients = blockDCTL0(block, k)
    % Extract the 'k' Fourier basis vectors withthe top projection
    % coefficients.
    
    [M, N] = size(block);
    
    % Unravel the block into a single column vector.
    block_vector = reshape(block, M * N, 1);
    
    % Compute the FFT.
    dct_coefficients = dct(block_vector);
    
    % Record the top 'k' coefficients.
    [sorted, indices] = sort(dct_coefficients, 'descend');
    coefficients = dct_coefficients;
    coefficients(indices(k:end)) = 0;
end