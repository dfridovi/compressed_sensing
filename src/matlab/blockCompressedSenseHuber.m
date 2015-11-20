function coefficients = ...
    blockCompressedSenseHuber(block, rho, alpha, basis, mixing)
    % Return reversed Huber compressed sensing result given rho, 
    % alpha, basis, and mixing.
    
    [M, N] = size(block);
    
    % Unravel the block into a single column vector.
    block_vector = reshape(block, M * N, 1);
    
    % Resample the image according to the mixing matrix.
    block_measured = mixing * block_vector;
    
    % Construct the problem and solve.
    cvx_begin quiet
    variable coefficients(M * N)
    minimize(sum_square(basis * coefficients - block_measured) + ... 
             2 * alpha * sum(berhu(rho * coefficients / sqrt(alpha), 1)))
    cvx_end
end