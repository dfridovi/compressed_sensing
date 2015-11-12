"""
Test script. Run compressed sensing on a 2D image in image domain and see what we get.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import  misc
import BasicFunctions as bf
import Sketching as sketch

# Parameters.
IMAGE_PATH = "../../data/"
IMAGE_NAME = "lenna.png"
SIZE = (50, 50)
ALPHA = 100.0
BASIS_OVERSAMPLING = 1.0

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE).astype(np.float32)

# Obtain Fourier basis.
basis, coefficients = sketch.basisCompressedSenseImgL1(img, ALPHA, BASIS_OVERSAMPLING)

# Compute reconstruction.
reconstruction = (basis * coefficients).reshape(img.shape)
    
# print estimate of sparsity
print np.median(np.asarray(coefficients.T))

# Plot.
max_value = np.absolute(coefficients).max()
plt.figure(1)
plt.subplot(121)
plt.imshow(reconstruction, cmap="gray")
plt.title("Reconstruction using image basis \n %.2f%% sparsity" %
           (100.0-((np.absolute(coefficients) > 0.01*max_value).sum()*100.0/(SIZE[0]*SIZE[1]))))


plt.subplot(122)
plt.hist(np.absolute(coefficients), bins=len(coefficients) / 50)
plt.title("Sparsity Histogram, alpha = %1.1f" %ALPHA)
plt.xlabel("Coefficient Magnitude")
plt.ylabel("Number of Coefficients")
plt.show()


