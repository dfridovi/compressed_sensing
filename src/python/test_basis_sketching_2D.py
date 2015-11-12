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
SIZE = (50, 50)
ALPHA = 2
BASIS_OVERSAMPLING = 1.0

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE)

# Obtain Fourier basis.
basis, coefficients = sketch.basisSketchL1(img, ALPHA, BASIS_OVERSAMPLING)

# Compute reconstruction.
reconstruction = (basis * coefficients).reshape(img.shape)
    
# Plot.
plt.figure(1)
plt.subplot(121)
plt.imshow(reconstruction, cmap="gray")

max_value = np.absolute(coefficients).max()
plt.title("Reconstruction using random basis \n in image domain \n %.2f%% sparsity" %
           (100.0-((np.absolute(coefficients) > 0.01 * max_value).sum()*100.0/(SIZE[0]*SIZE[1]))))



ax = plt.subplot(122)
plt.hist(np.absolute(coefficients), bins=len(coefficients) / 50)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, end/4))
plt.xlabel("Coefficient Magnitude")
plt.ylabel("Number of Coefficients")
plt.title("Sparsity Histogram, alpha = %.1f" %ALPHA)
plt.show()
