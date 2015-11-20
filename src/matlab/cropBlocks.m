function [ cropped_blocks ] = getBlocks( padded_blocks, original_block_size )
% cropBlocks( padded_blocks, original_block_size )
% Crop a set of padded blocks.

    % Throw an error if 'padded_blocks' is not a 3D array.
    if numel(size(blocks)) ~= 3
        error('Padded blocks is not a 3D array.');
    end
    
    % Extract dimensions.
    [M, N, B] = size(padded_blocks);
    offset = [round(0.5 * (M - original_block_size(1))), ...
	      round(0.5 * (N - original_block_size(2)))];
    
    % Pad image, check new shape
    cropped_blocks = zeros(original_block_size);

    % Iterate through the image and append to 'blocks.'
    for b = 1:B
       padded_block = padded_blocks(:, :, b);
       cropped_block = padded_block(offset(1) + 1:original_block_size(1) + offset(1), ...
				    offset(2) + 1:original_block_size(2) + offset(2));
       cropped_blocks(:, :, b) = cropped_block;
    end

end

