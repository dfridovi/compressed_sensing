function basis = computeDCTBasis(M, N)
    % Compute a Fourier basis matrix in N dimensions.

    basis = zeros(M * N, M * N);
    for i = 1:M * N
       
        % Set up a dummy vector with only one index high.
        dummy_vector = zeros(M * N, 1);
        dummy_vector(i) = 1;
        
        % Take the IFFT.
        basis_vector = idct2(reshape(dummy_vector, M, N));
        
        % Append to the basis matrix.
        basis(:, i) = reshape(basis_vector, M * N, 1);
    end
end
