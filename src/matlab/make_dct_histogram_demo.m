% Read in an image. Compute the 2D DCT coefficients. Make a histogram.

img = rgb2gray(imread('../../data/campanile.jpg'));
% figure;
% imshow(img);

coefs = dct2(double(img));
% figure;
% hist(reshape(coefs, numel(coefs), 1), 1000)

figure;
%coefs = dct(reshape(double(img), numel(img), 1));
%plot(coefs)
imagesc(log(1 + abs(coefs)));