"""
Test script. Run compressed sensing on a 2D image and see what we get.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import  misc
import BasicFunctions as bf
import Sketching as sketch

# Parameters.
IMAGE_PATH = "../../data/"
IMAGE_NAME = "lenna.png"
SIZE = (30, 30)
ALPHA = 1.0
BASIS_OVERSAMPLING = 1.0

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE)

# Obtain Fourier basis.
basis, coefficients = sketch.basisSketchL1(img, ALPHA, BASIS_OVERSAMPLING)

# Compute reconstruction.
reconstruction = (basis * coefficients).reshape(img.shape)
    
# Plot.
plt.figure(1)
plt.imshow(reconstruction, cmap="gray")
plt.title(("Reconstruction using %d random basis vectors in image domain." %
           (np.absolute(coefficients) > 1e-8).sum()))
plt.show()
