function [dct_basis, block_coefficients] = ...
    compressedSenseDCTHuber(blocks, rho, alpha, basis_oversampling)
% Sketch the image blocks in the DCT domain. Procedure:
%     1. Choose a random matrix to mix the DCT components.
%     2. Solve the L1-penalized least-squares problem.
% 
%     min_x ||AFx - m||_2^2 + 2 * alpha * B(rho * x / sqrt(alpha))
%     
%     where: m = sampled image,
%            x = representation,
%            A = mixing matrix,
%            F = DCT basis
%            B = reversed Huber function

    [M, N, B] = size(blocks); % blocks is a 3D array
    
    % Generate a mixing matrix.
    mixing = randn(round(M * N * basis_oversampling), M * N);
    
    % Generate the DCT basis.
    dct_basis = computeDCTBasis(M, N);
    
    % Pre-mulitply the dct basis by the mixing matrix.
    basis_premultiplied = mixing * dct_basis;
    
    % Iterate over all blocks. Store coefficients in a 2D array.
    block_coefficients = zeros(M * N, B);
    parfor i = 1:B
       block = blocks(:, :, i);
       %fprintf('Working on block %d of %d...\n', i, B);
       coefficients = blockCompressedSenseHuber(block, rho, alpha, ...
                                                basis_premultiplied, ...
                                                mixing);
       block_coefficients(:, i) = coefficients;
    end
end