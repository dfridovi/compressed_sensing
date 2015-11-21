function [ padded_blocks ] = padBlocks( blocks, pad_percent )
% padBlocks( blocks, pad_percent )
% Pad each block.

    % Throw an error if 'blocks' is not a 2D array.
    if numel(size(blocks)) ~= 2
        error('Blocks is not a 2D array.');
    end
    
    % Extract dimensions.
    [M, B] = size(blocks);
    pad_size = round(M * pad_percent);
    
    % Pad image, check new shape
    padded_blocks = zeros(M + 2*pad_size, B);
    
    % Iterate through the image and append to 'blocks.'
    for b = 1:B
       block = blocks(:, b);
       padded_block = padarray(block, pad_size, 'symmetric');
       padded_blocks(:, b) = padded_block; 
    end

end

