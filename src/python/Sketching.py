"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

from functools import partial
import sys

import numpy as np
from scipy import fftpack

import cvxpy as cvx


import BasicFunctions as bf

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

def blockFourierL0(block, k):
    """ Extract the 'k' Fourier basis vectors with the top projection coefficients. """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(block.ravel()).T

    # Compute the FFT.
    fourier = np.fft.fft(img_vector)

    # Record the top 'k' coefficients.
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier))
    coefficients = fourier
    coefficients[sorted_indices[k:]] = 0.0
    coefficients = np.asmatrix(coefficients).T
    
    return coefficients

def basisFourierL0(blocks, k):
    """ Run blockwise top 'k' Fourier compresssion."""

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]
    
    # Generate DCT basis and premultiply image.
    fourier_basis = computeFourierBasis(block_len)
    
    # Make a special function given these parameters.
    print "Creating a partial function."
    blockFourier = partial(blockFourierL0, k=k)
    
    # Run compressed sensing on each block and store results.
    print "Running CS on the pool."
    block_coefficients = map(blockFourier, blocks)

    return fourier_basis, block_coefficients

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
        
def blockDCTL0(block, k):
    """ Extract the 'k' DCT basis vectors with the top projection coefficients. """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(block.ravel()).T

    # Compute the DCT.
    dct = fftpack.dct(img_vector).astype(np.float32)

    # Record the top 'k' coefficients.
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier))
    coefficients = dct
    coefficients[sorted_indices[k:]] = 0.0
    coefficients = np.asmatrix(coefficients).T
    
    return coefficients

def basisFourierL0(blocks, k):
    """ Run blockwise top 'k' DCT compresssion."""

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]
    
    # Generate DCT basis and premultiply image.
    dct_basis = computeDCTBasis(block_len)
    
    # Make a special function given these parameters.
    print "Creating a partial function."
    blockDCT = partial(blockDCTL0, k=k)
    
    # Run compressed sensing on each block and store results.
    print "Running CS on the pool."
    block_coefficients = map(blockDCT, blocks)

    return dct_basis, block_coefficients

def blockCompressedSenseL1(block, rho, alpha, basis_premultiplied, mixing_matrix):
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
    REG = cvx.sum_squares(coefficients)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + rho*REG + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)
    
    # Solve.
    problem.solve(verbose=False, solver='SCS')
    
    # Print problem status.
    print "Problem status: " + str(problem.status)
    sys.stdout.flush()

    return coefficients.value

def basisCompressedSenseDCTL1(blocks, rho, alpha, basis_oversampling=1.0):
    """
    Sketch the image blocks in the DCT domain. Procedure: 
    1. Choose a random matrix to mix the DCT components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||AFx - m||_2^2 + rho * ||x||_2^2 + alpha * ||x||_1, where y = image, 
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

    # Make a special function given these parameters.
    print "Creating a partial function."
    blockCS = partial(blockCompressedSenseL1,
                      rho=rho,
                      alpha=alpha,
                      basis_premultiplied=basis_premultiplied,
                      mixing_matrix=mixing_matrix)
    
    # Run compressed sensing on each block and store results.
    print "Running CS on the pool."
    block_coefficients = map(blockCS, blocks)

    return dct_basis, block_coefficients

def blockCompressedSenseHuber(block, rho, alpha, basis_premultiplied, mixing_matrix):
    """ Run reversed Huber compressed sensing given alpha and a basis."""

    # Get block size.
    block_len = block.shape[0] * block.shape[1]
    
    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(block.ravel()).T
    
    # Determine m (samples)
    img_measured = mixing_matrix * img_vector
    
    # Construct the problem.
    coefficients = cvx.Variable(block_len)
    coefficients_premultiplied = basis_premultiplied * coefficients

    huber_penalty = reversed_huber(rho * coefficients / np.sqrt(alpha))

    L2 = cvx.sum_squares(coefficients_premultiplied - img_measured)
    RH = cvx.sum_entries(huber_penalty)
    objective = cvx.Minimize(L2 + 2*alpha*RH)
    constraints = []
    problem = cvx.Problem(objective, constraints)
    
    # Solve.
    problem.solve(verbose=False, solver='SCS')
    
    # Print problem status.
    print "Problem status: " + str(problem.status)
    sys.stdout.flush()

    return coefficients.value

def basisCompressedSenseDCTHuber(blocks, rho, alpha, basis_oversampling=1.0):
    """
    Sketch the image blocks in the DCT domain. Procedure: 
    1. Choose a random matrix to mix the DCT components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||AFx - m||_2^2 + rho * 2 * alpha * B(rho * x / sqrt(alpha)), where y = image, 
                                                                     x = representation, 
                                                                     A = mixing matrix,
                                                                     F = DCT basis
                                                                     m = Ay
    B = reversed Huber function
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

    # Make a special function given these parameters.
    blockHuber = partial(blockCompressedSenseHuber,
                         rho=rho,
                         alpha=alpha,
                         basis_premultiplied=basis_premultiplied,
                         mixing_matrix=mixing_matrix)
    
    # Run compressed sensing on each block and store results.
    block_coefficients = map(blockHuber, blocks)

    return dct_basis, block_coefficients

def basisCompressedSenseImgL1(img, rho, alpha, basis_oversampling=1.0):
    """
    Sketch the image in the image domain. Procedure: 
    1. Choose a random matrix to mix the image domain basis components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||Ax - m||_2^2 + rho * ||x||_2^2 + alpha * ||x||_1, where y = image, 
                                                                    x = representation, 
                                                                    A = mixing matrix,
                                                                    m = Ay
    """

    # Get block size.
    block_len = blocks[0].shape[0] * blocks[0].shape[1]
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(int(block_len * basis_oversampling),
                                    block_len)
    
    # Make a special function given these parameters.
    blockCS = partial(blockCompressedSenseL1,
                      rho=rho,
                      alpha=alpha,
                      basis_premultiplied=mixing_matrix,
                      mixing_matrix=mixing_matrix)
    
    # Run compressed sensing on each block and store results.
    block_coefficients = map(blockCS, blocks)

    return np.identity(len(img_vector)), block_coefficients

def computeSparsity(block_coefficients):
    """ Compute total sparsity given the list of block-wise coefficients."""

    # Get block size.
    size = len(block_coefficients[0])
    
    # Running list of sparsity coefficients.
    sparsity = []
    for coefficients in block_coefficients:
        max_value = np.absolute(coefficients).max() 
        sparsity.append(100.0-(((np.absolute(coefficients) >
                                 0.01*max_value).sum())*100.0 / size))

    return sparsity
        
def visualizeBlockwiseSparsity(blocks, sparsity, original_shape):
    """ Visualize blockwise sparsity."""

    blocks = np.array(blocks)
    sparsity = np.array(sparsity)
    new_image = np.zeros(original_shape)
    k = blocks[0].shape[0]
    n_vert = original_shape[0] / k
    n_horiz = original_shape[1] / k

    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            new_image[i*k:(i+1)*k,
                      j*k:(j+1)*k] = bf.adjustExposure(blocks[n_horiz*i + j],
                                                       1.0 - 0.01*sparsity[n_horiz*i + j])

    return new_image


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
