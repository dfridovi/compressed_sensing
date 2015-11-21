function [ cropped_blocks ] = cropBlocks( padded_blocks, original_block_size )
% cropBlocks( padded_blocks, original_block_size )
% Crop a set of padded blocks.

    % Throw an error if 'padded_blocks' is not a 2D array.
    if numel(size(padded_blocks)) ~= 2
        error('Padded blocks is not a 2D array.');
    end
    
    % Extract dimensions.
    [M, B] = size(padded_blocks);
    offset = round(0.5 * (M - original_block_size(1)));
    
    % Pad image, check new shape
    cropped_blocks = zeros(original_block_size, B);

    % Iterate through the image and append to 'blocks.'
    for b = 1:B
       padded_block = padded_blocks(:, b);
       cropped_block = padded_block(offset + 1:original_block_size + offset);
       cropped_blocks(:, b) = cropped_block;
    end

end

