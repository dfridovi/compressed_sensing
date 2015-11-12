"""
Test script. Run compressed sensing on a 2D image in Fourier domain and see what we get.
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
ALPHA = 1.5
BASIS_OVERSAMPLING = 1.0

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE).astype(np.float32)

# Obtain Fourier basis.
basis, coefficients = sketch.basisSketchDCTL1(img, ALPHA, BASIS_OVERSAMPLING)

# Compute reconstruction.
reconstruction = (basis * coefficients).reshape(img.shape)
    
# Plot.
max_value = np.absolute(coefficients).max()
plt.figure(1)
plt.subplot(121)
plt.imshow(reconstruction, cmap="gray")
plt.title(("Reconstruction using \n %d random basis vectors \n in DCT domain." %
           (np.absolute(coefficients) > 0.01 * max_value).sum()))


plt.subplot(122)
plt.hist(np.absolute(coefficients), bins=len(coefficients) / 50)
plt.title("Sparsity Histogram, alpha = %1.1f" %ALPHA)
plt.xlabel("Coefficient")
plt.ylabel("Magnitude")
plt.show()
