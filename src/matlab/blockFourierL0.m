function coefficients = blockFourierL0(block, k)
    % Extract the 'k' Fourier basis vectors withthe top projection
    % coefficients.
    
    [M, N] = size(block);
    
    % Unravel the block into a single column vector.
    block_vector = reshape(block, M * N, 1);
    
    % Compute the FFT.
    fourier_coefficients = fft(block_vector);
    
    % Record the top 'k' coefficients.
    [sorted, indices] = sort(fourier_coefficients, 'descend');
    coefficients = fourier_coefficients;
    coefficients(indices(k:end)) = 0;
end