function [ blocks ] = getBlocks( img, k, overlap )
% getBlocks( img, k, overlap_percent )
% Break the image up into kx1 blocks. Crop if necessary.

    % Throw an error if not grayscale.
    if numel(size(img)) ~= 2
        error('Image is not grayscale. Returning empty block list.')
    end
    img_vector = reshape(img, numel(img), 1);
    padded_img_vector = padarray(img_vector, round(overlap * k), 'symmetric');
    B = floor(numel(img_vector) / k);
    M = k+2*round(overlap*k);

    window = bartlett(M);
    blocks = zeros(M,B);
    for i = 0:B-1
        blocks(:,i+1) = double(padded_img_vector(i*k+1:i*k+M)).* window;
    end
end
