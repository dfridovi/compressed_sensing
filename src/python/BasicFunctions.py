"""
Helper functions for compressed_sensing.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def imread(imfile):
    """ Read image from file and normalize. """
    
    img = mpimg.imread(imfile).astype(np.float32)
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

def truncate(img):
    """ Truncate values in image to range [0.0, 1.0]. """
    
    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    return img

def rescale(img):
    """ Rescale image values linearly to the range [0.0, 1.0]. """
    
    return (img - img.min()) / (img.max() - img.min())

def rgb2gray(img):
    """ Convert an RGB image to grayscale. """
    
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    
    return 0.299*r + 0.587*g + 0.114*b

def bgr2gray(img):
    """ Convert a BGR image to grayscale. """
    
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    
    return 0.299*r + 0.587*g + 0.114*b

def adjustExposure(img, gamma=1.0):
    """ Simulate changing the exposure by scaling the image intensity."""
    
    return np.power(img, gamma)
