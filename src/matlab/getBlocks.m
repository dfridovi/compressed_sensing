function [ blocks ] = getBlocks( img, k, overlap_percent )
% getBlocks( img, k, overlap_percent )
% Break the image up into kxk blocks. Crop if necessary.

    % Throw an error if not grayscale.
    if numel(size(img)) ~= 2
        error('Image is not grayscale. Returning empty block list.')
    end
    
    overlap = int(k*overlap_percent);

    
    n_vert = int(size(img,1) / k);
    n_horiz = int(size(img,2) / k);
    
    blocks = zeros(k+2*overlap, k+2*overlap, n_vert*n_horiz);

    % Pad image, check new shape
    padded_img = padarray(img, [overlap, overlap], 'replicate');

    % Iterate through the image and append to 'blocks.'
    for i = 0:n_vert-1
        for j = 0:n_horiz-1
            blocks(:,:,n_horiz*i+j+1) = padded_img(i*k+1:((i+1)*k+2*overlap)+1,...
                j*k+1:((j+1)*k+2*overlap)+1);
        end
    end


end

