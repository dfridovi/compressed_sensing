function basis = computeFourierBasis(N)
    % Compute a Fourier basis matrix in N dimensions.

    basis = zeros(N, N);
    for i = 1:N
       
        % Set up a dummy vector with only one index high.
        dummy_vector = zeros(N, 1);
        dummy_vector(i) = 1;
        
        % Take the IFFT.
        basis_vector = ifft(dummy_vector);
        
        % Append to the basis matrix.
        basis(:, i) = basis_vector;
    end
end
