function [ padded_blocks ] = getBlocks( blocks, pad_percent )
% padBlocks( blocks, pad_percent )
% Pad each block.

    % Throw an error if 'blocks' is not a 3D array.
    if numel(size(blocks)) ~= 3
        error('Blocks is not a 3D array.');
    end
    
    % Extract dimensions.
    [M, N, B] = size(blocks);
    pad_size = [round(M * pad_percent), round(N * pad_percent)];
    
    % Pad image, check new shape
    padded_blocks = zeros(M + 2*pad_size(1), N + 2*pad_size(2), B);

    % Iterate through the image and append to 'blocks.'
    for b = 1:B
       block = blocks(:, :, b);
       padded_block = padarray(block, pad_size, 'symmetric');
       padded_blocks(:, :, b) = padded_block;
    end

end

