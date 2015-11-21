function [ new_image ] = assembleBlocks( blocks, block_size, original_shape, overlap )
% assembleBlocks( blocks, original_shape, overlap )
% Reassemble the image from a list of blocks.
    
    [M, B] = size(blocks);
    img_vector = zeros(original_shape(1)*original_shape(2)+2*round(overlap*block_size),1);
    for i=0:B-1
        img_vector(i*block_size+1:i*block_size+M) = ...
            img_vector(i*block_size+1:i*block_size+M) + ...
            blocks(:,i+1);
    end
    new_image = reshape(img_vector(round(overlap*block_size)+1:round(overlap*block_size)...
        +original_shape(1)*original_shape(2)), original_shape);
end

