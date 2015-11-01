"""
Helper functions for compressed_sensing.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def imread(imfile):
    """ Read image from file and normalize. """
    
    img = mpimg.imread(imfile).astype(np.float)
    img = rescale(img)
    return img

def imshow(img, title="", cmap="gray", cbar=False):
    """ Show image to screen. """
    
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    
    if cbar:
        plt.colorbar()
        plt.show()
        
def imsave(img, imfile):
    """ Save image to file."""
    
    mpimg.imsave(imfile, img)
