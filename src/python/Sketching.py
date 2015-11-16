"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

from multiprocessing import Pool
from functools import partial
import sys

import numpy as np
from scipy import fftpack
import cvxpy as cvx

def computeFourierBasis(N):
    """ Compute a Fourier basis matrix in N dimensions. """

    basis = np.zeros((N, N)) + 0.0j
    for i in range(N):

        # Set up a dummy vector with only one index high.
        dummy_vector = np.zeros(N)
        dummy_vector[i] = 1.0

        # Take the IFFT.
        basis_vector = np.fft.ifft(dummy_vector)

        # Append to basis matrix.
        basis[:, i] = basis_vector

    return np.asmatrix(basis)
        
def basisFourier(img, k):
    """ Extract the 'k' Fourier basis vectors with the top projection coefficients. """

    # Unravel this image into a single column vector.
    img_vector = img.ravel()

    # Compute the FFT.
    fourier = np.fft.fft(img_vector)
    
    # Record the top 'k' coefficients.
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier))
    coefficients = fourier
    coefficients[sorted_indices[k:]] = 0.0
    coefficients = np.asmatrix(coefficients).T
    
    # Generate basis matrix for these indices.
    basis = computeFourierBasis(len(coefficients))
    
    return basis, coefficients

def computeDCTBasis(N):
    """ Compute a DCT basis matrix in N dimensions. """

    basis = np.zeros((N, N), dtype=np.float32)
    for i in range(N):

        # Set up a dummy vector with only one index high.
        dummy_vector = np.zeros(N)
        dummy_vector[i] = 1.0

        # Take the IFFT.
        basis_vector = fftpack.idct(dummy_vector)

        # Append to basis matrix.
        basis[:, i] = basis_vector.astype(np.float32)

    return np.asmatrix(basis)
        
def basisDCT(img, k):
    """ Extract the 'k' DCT basis vectors with the top projection coefficients. """

    # Unravel this image into a single column vector.
    img_vector = img.ravel()

    # Compute the FFT.
    dct = fftpack.dct(img_vector).astype(np.float32)
    
    # Record the top 'k' coefficients.
    sorted_indices = np.argsort(-1.0 * np.absolute(dct))
    coefficients = dct
    coefficients[sorted_indices[k:]] = 0.0
    coefficients = np.asmatrix(coefficients).T
    
    # Generate basis matrix for these indices.
    basis = computeDCTBasis(len(coefficients))
    
    return basis, coefficients

def basisSketchL1(img, alpha, basis_oversampling=1.0):
    """
    Sketch the image. Procedure: 
    1. Choose a random basis.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||x - A^+ y||_2^2 + alpha * ||x||_1 : y = image, 
                                               x = representation, 
                                               A = mixing basis (A^+ is pseudoinverse)
    """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(img.ravel()).T
    
    # Generate a random basis.
    basis = np.random.randn(len(img_vector),
                            int(len(img_vector) * basis_oversampling))
    basis_pinv = np.linalg.pinv(basis)

    # Pre-multiply image by basis pseudoinverse.
    img_premultiplied = basis_pinv * img_vector

    # Construct the problem.
    coefficients = cvx.Variable(basis.shape[1])
    L2 = cvx.sum_squares(coefficients - img_premultiplied)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)

    # Solve.
    problem.solve(verbose=True, solver='SCS')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return basis, coefficients.value

def blockCompressedSenseL1(block, alpha, basis_premultiplied, mixing_matrix):
    """ Run L1 compressed sensing given alpha and a basis."""

    # Get block size.
    block_len = block.shape[0] * block.shape[1]
    
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
    print "Problem status: " + str(problem.status)
    sys.stdout.flush()

    return coefficients.value

def basisCompressedSenseDCTL1(blocks, alpha, basis_oversampling=1.0, num_processors=4):
    """
    Sketch the image blocks in the DCT domain. Procedure: 
    1. Choose a random matrix to mix the DCT components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||AFx - m||_2^2 + alpha * ||x||_1, where y = image, 
                                                   x = representation, 
                                                   A = mixing matrix,
                                                   F = DCT basis
                                                   m = Ay
    """

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(int(block_len * basis_oversampling),
                                    block_len)
    
    # Generate DCT basis and premultiply image.
    dct_basis = computeDCTBasis(block_len)
    
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

def basisCompressedSenseImgL1(img, alpha, basis_oversampling=1.0):
    """
    Sketch the image in the image domain. Procedure: 
    1. Choose a random matrix to mix the DCT components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||Ax - m||_2^2 + alpha * ||x||_1, where y = image, 
                                                   x = representation, 
                                                   A = mixing matrix,
                                                   F = DCT basis (not applicable)
                                                   m = Ay
    """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(img.ravel()).T
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(int(len(img_vector) * basis_oversampling),
        len(img_vector))


    # Determine m (samples)
    img_measured = mixing_matrix * img_vector

    # Construct the problem.
    coefficients = cvx.Variable(len(img_vector))
    L2 = cvx.sum_squares(mixing_matrix*coefficients - img_measured)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)

    # Solve.
    problem.solve(verbose=True, solver='SCS')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return np.identity(len(img_vector)), coefficients.value

def getBlocks(img, k):
    """ Break the image up into kxk blocks. Crop if necessary."""

    # Throw an error if not grayscale.
    if len(img.shape) != 2:
        print "Image is not grayscale. Returning empty block list."
        return []
    
    blocks = []
    n_vert = img.shape[0] / k
    n_horiz = img.shape[1] / k

    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            blocks.append(img[i*k:(i+1)*k, j*k:(j+1)*k])

    return blocks

def assembleBlocks(blocks, original_shape):
    """ Reassemble the image from a list of blocks."""

    blocks = np.array(blocks)
    new_image = np.zeros(original_shape)
    k = blocks[0].shape[0]
    n_vert = original_shape[0] / k
    n_horiz = original_shape[1] / k

    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            new_image[i*k:(i+1)*k, j*k:(j+1)*k] = blocks[n_horiz*i + j]

    return new_image
