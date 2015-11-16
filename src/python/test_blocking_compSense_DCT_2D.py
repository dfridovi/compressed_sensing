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
NUM_PROCESSORS = 6
IMAGE_PATH = "../../data/"
IMAGE_NAME = "lenna.png"
BLOCK_SIZE = 30
ALPHA = 1.0
BASIS_OVERSAMPLING = 1.0

"""
def basisCompressedSenseDCTL1(blocks, alpha, basis_oversampling=1.0, num_processors=4):

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(int(block_len * basis_oversampling),
                                    block_len)
    
    # Generate DCT basis and premultiply image.
    dct_basis = sketch.computeDCTBasis(block_len)
    
    # Pre-multiply image by basis mixing matrix (AF)
    basis_premultiplied = mixing_matrix * dct_basis.T

    # Make a processor pool.
    print "Creating a processor pool." 
    pool = Pool(num_processors)

    # Make a special function given these parameters.
    print "Creating a partial function."
    blockCS = partial(blockCompressedSenseL1,
                      alpha=alpha,
                      basis_premultiplied=basis_premultiplied,
                      mixing_matrix=mixing_matrix)
    
    # Run compressed sensing on each block and store results.
    print "Running CS on the pool."
    block_coefficients = map(blockCS, blocks)

    return dct_basis, block_coefficients

def blockCompressedSenseL1(block, alpha, basis_premultiplied, mixing_matrix):

    sys.stdout.write("Running compressed sensing on this block.\n")
    sys.stdout.flush()

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(block.ravel()).T
    
    # Determine m (samples)
    img_measured = mixing_matrix * img_vector
    
    # Construct the problem.
    coefficients = cvx.Variable(block_len)
    coefficients_premultiplied = basis_premultiplied * coefficients
    L2 = cvx.sum_squares(coefficients_premultiplied - img_measured)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)
    
    # Solve.
    problem.solve(verbose=False, solver='SCS')
    
    # Print problem status.
    sys.stdout.write("Problem status: " + str(problem.status) + "\n")
    sys.stdout.flush()
"""

if __name__ == "__main__":

    # Import the image.
    img = misc.imresize(bf.rgb2gray(bf.imread(IMAGE_PATH + IMAGE_NAME)), (60, 60)).astype(np.float32)
    
    # Get blocks.
    blocks = sketch.getBlocks(img, BLOCK_SIZE)
    print "Got %d blocks." % len(blocks)
    
    
    # Compress each block.
    print "Running CS on each block..."
    basis, block_coefficients = sketch.basisCompressedSenseDCTL1(blocks,
                                                                 ALPHA,
                                                                 BASIS_OVERSAMPLING,
                                                                 NUM_PROCESSORS)
    
    # Compute reconstruction for each block.
    print "Reconstructing..."
    reconstructed_blocks = []
    for i, coefficients in enumerate(block_coefficients):
        print "Progress: %d / %d" % (i, len(block_coefficients))    
        reconstructed_blocks.append((basis * coefficients).reshape((BLOCK_SIZE,
                                                                    BLOCK_SIZE)))
        
    # Reassemble.
    reconstruction = sketch.assembleBlocks(reconstructed_blocks, img.shape)
    
    # print estimate of sparsity
    #print np.median(np.asarray(coefficients.T))
    
    # Plot.
    #max_value = np.absolute(coefficients).max()
    plt.figure(1)
    #plt.subplot(121)
    plt.imshow(reconstruction, cmap="gray")
    #plt.title("Reconstruction using DCT basis \n %.2f%% sparsity" %
    #           (100.0-((np.absolute(coefficients) > 0.01*max_value).sum()*100.0/(SIZE[0]*SIZE[1]))))
    plt.show()
    """
    plt.subplot(122)
    plt.hist(np.absolute(coefficients), bins=len(coefficients) / 50)
    plt.title("Sparsity Histogram, alpha = %1.1f" %ALPHA)
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Number of Coefficients")
    plt.show()
    
    
    """
