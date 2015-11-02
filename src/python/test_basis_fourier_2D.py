"""
Test script. Run Fourier decomposition on a 2D image and see what we get.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import  misc
import BasicFunctions as bf
import Sketching as sketch

# Parameters.
IMAGE_PATH = "../../data/"
IMAGE_NAME = "lenna.png"
SIZE = (100, 100)
K = 100

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE)

# Obtain Fourier basis.
basis, coefficients = sketch.basisFourier(img, K)

# Compute reconstruction.
reconstruction = np.zeros(img.shape) + 0j
for i in range(K):
    component = basis[:,i] * coefficients[i]
    reconstruction += component.reshape(img.shape)

reconstruction = np.absolute(reconstruction)
    
# Plot.
plt.figure(1); plt.imshow(img, cmap="gray"); plt.show()
plt.figure(2); plt.imshow(reconstruction, cmap="gray"); plt.show()
