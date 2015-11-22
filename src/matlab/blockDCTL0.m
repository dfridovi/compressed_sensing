function coefficients = blockDCTL0(block, k)
    % Extract the 'k' Fourier basis vectors withthe top projection
    % coefficients.
        
    % Compute the FFT.
    dct_coefficients = reshape(dct2(block), numel(block), 1);
    
    % Record the top 'k' coefficients.
    [sorted, indices] = sort(abs(dct_coefficients), 'descend');
    coefficients = dct_coefficients;
    coefficients(indices(floor(k)+1:end)) = 0;
end