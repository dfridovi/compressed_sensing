function [ blocks ] = getBlocks( img, k, overlap_percent )
% getBlocks( img, k, overlap_percent )
% Break the image up into kx1 blocks. Crop if necessary.

    % Throw an error if not grayscale.
    if numel(size(img)) ~= 2
        error('Image is not grayscale. Returning empty block list.')
    end
    
    overlap = round(k*overlap_percent);

    
    n_vert = floor(size(img,1) / k);
    n_horiz = floor(size(img,2) / k);
    
    blocks = zeros(k+2*overlap, k+2*overlap, n_vert*n_horiz);

    % Pad image, check new shape
    padded_img = padarray(img, [overlap, overlap], 'replicate');

    % Iterate through the image and append to 'blocks.'
    for i = 0:n_vert-1
        for j = 0:n_horiz-1
            blocks(:,:,n_horiz*i+j+1) = padded_img(i*k+1:((i+1)*k+2*overlap),...
                j*k+1:((j+1)*k+2*overlap));
        end
    end


end