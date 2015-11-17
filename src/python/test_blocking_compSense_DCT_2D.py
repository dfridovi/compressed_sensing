"""
Test script. Run compressed sensing on a 2D image in DCT domain with blocking.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import  misc
import BasicFunctions as bf
import Sketching as sketch

from multiprocessing import Pool
from functools import partial
import sys
import cvxpy as cvx

# Parameters.
IMAGE_PATH = "../../data/"
IMAGE_NAME = "lenna.png"
BLOCK_SIZE = 10
ALPHA = 1.0
BASIS_OVERSAMPLING = 1.0
OVERLAP_PERCENT = 0.5;

if __name__ == "__main__":

    # Import the image.
    img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), (40, 40)).astype(np.float32)

    
    # Get blocks.
    blocks = sketch.getBlocks_withOverlap(img, BLOCK_SIZE, OVERLAP_PERCENT)
    print "Got %d blocks." % len(blocks)
    
    
    # Compress each block.
    print "Running CS on each block..."
    basis, block_coefficients = sketch.basisCompressedSenseDCTL1(blocks,
                                                                 ALPHA,
                                                                 BASIS_OVERSAMPLING)

    # Get sparsity.
    sparsity = sketch.computeSparsity(block_coefficients)
    print "Sparsity: " + str(sparsity)
    
    # Compute reconstruction for each block.
    print "Reconstructing..."
    reconstructed_blocks = []
    for i, coefficients in enumerate(block_coefficients):
        print "Progress: %d / %d" % (i, len(block_coefficients))    
        reconstructed_blocks.append((basis * coefficients).reshape((blocks[0].shape[0],
                                                                    blocks[0].shape[1])))
        
    # Reassemble.
    reconstruction = sketch.assembleBlocks_withOverlap(reconstructed_blocks, 
                                                        BLOCK_SIZE, img.shape, OVERLAP_PERCENT)

    # Note this may not work because the reconstructed_blocks are bigger than expected
    # because of the overlap
    visualization = sketch.visualizeBlockwiseSparsity(reconstructed_blocks,
                                                      sparsity,
                                                      img.shape)

    
    # print estimate of sparsity
    #print np.median(np.asarray(coefficients.T))
    
    # Plot.
    #max_value = np.absolute(coefficients).max()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(reconstruction, cmap="gray")
    plt.colorbar()
    #plt.title("Reconstruction using DCT basis \n %.2f%% sparsity" %
    #           (100.0-((np.absolute(coefficients) > 0.01*max_value).sum()*100.0/(SIZE[0]*SIZE[1]))))

    plt.subplot(122)
    plt.imshow(visualization, cmap="gray")
    plt.colorbar()
    plt.show()
    
    """
    plt.subplot(122)
    plt.hist(np.absolute(coefficients), bins=len(coefficients) / 50)
    plt.title("Sparsity Histogram, alpha = %1.1f" %ALPHA)
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Number of Coefficients")
    plt.show()
    
    
    """
