function coefficients = ...
    blockCompressedSenseL1(block, alpha, basis, mixing)
    % Return L1 compressed sensing result given rho, alpha, 
    % basis, and mixing.
    
    [M, N] = size(block);
    
    % Unravel the block into a single column vector.
    block_vector = reshape(block, M * N, 1);
    
    % Resample the image according to the mixing matrix.
    block_measured = mixing * block_vector;
    
    % Construct the problem and solve.
    cvx_begin quiet
    variable coefficients(M * N)
    minimize( sum_square( basis * coefficients - block_measured ) + ... 
              alpha * norm( coefficients, 1 ) )
    cvx_end
   % [coefficients, info] = lasso(basis, block_measured, 'Lambda', alpha);
  %  beta0 = info.Intercept;
    
end