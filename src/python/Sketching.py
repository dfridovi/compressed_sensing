"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

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

def basisSketchDCTL1(img, alpha, basis_oversampling=1.0):
    """
    Sketch the image in the DCT domain. Procedure: 
    1. Choose a random matrix to mix the DCT components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||x - A^+ F^T y||_2^2 + alpha * ||x||_1, where y = image, 
                                                   x = representation, 
                                                   A = mixing matrix,
                                                   F = DCT basis
    """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(img.ravel()).T
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(len(img_vector),
                                    int(len(img_vector) * basis_oversampling))
    mixing_matrix_pinv = np.linalg.pinv(mixing_matrix)

    # Generate DCT basis and premultiply image.
    dct_basis = computeDCTBasis(len(img_vector))

    # Pre-multiply image by basis pseudoinverse.
    img_premultiplied = mixing_matrix_pinv * dct_basis.T * img_vector

    # Construct the problem.
    coefficients = cvx.Variable(mixing_matrix.shape[1])
    L2 = cvx.sum_squares(coefficients - img_premultiplied)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)

    # Solve.
    problem.solve(verbose=True, solver='SCS')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return dct_basis * mixing_matrix, coefficients.value
