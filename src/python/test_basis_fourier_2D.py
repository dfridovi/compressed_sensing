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
SIZE = (50, 50)
K = (SIZE[0]*SIZE[1]*0.75, SIZE[0]*SIZE[1]*0.5, SIZE[0]*SIZE[1]*0.25)

# Import the image.
img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), SIZE)

# Obtain Fourier basis.
basis, coefficients = sketch.basisFourier(img, K[0])

# Compute reconstruction.
reconstruction1 = (basis * coefficients).reshape(img.shape)
reconstruction1 = np.absolute(reconstruction1)

# Obtain Fourier basis.
basis, coefficients = sketch.basisFourier(img, K[1])

# Compute reconstruction.
reconstruction2 = (basis * coefficients).reshape(img.shape)
reconstruction2 = np.absolute(reconstruction2)

# Obtain Fourier basis.
basis, coefficients = sketch.basisFourier(img, K[2])

# Compute reconstruction.
reconstruction3 = (basis * coefficients).reshape(img.shape)
reconstruction3 = np.absolute(reconstruction3)
    
# Plot.
plt.figure(1)
plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.subplot(222)
plt.imshow(reconstruction1, cmap="gray")
plt.title("Fourier Reconstruction\nwith top %d%% of coefficients" 
	% ((K[0]*100.0)/(SIZE[0]*SIZE[1])))

plt.subplot(223)
plt.imshow(reconstruction2, cmap="gray")
plt.title("Fourier Reconstruction\nwith top %d%% of coefficients" 
	% ((K[1]*100.0)/(SIZE[0]*SIZE[1])))

plt.subplot(224)
plt.imshow(reconstruction3, cmap="gray")
plt.title("Fourier Reconstruction\nwith top %d%% of coefficients" 
	% ((K[2]*100.0)/(SIZE[0]*SIZE[1])))

plt.show() 



