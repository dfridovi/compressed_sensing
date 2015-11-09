"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

import numpy as np
import cvxpy as cvx

def basisFourier(img, k):
    """ Extract the 'k' Fourier basis vectors with the top projection coefficients. """

    # Unravel this image into a single column vector.
    img_vector = img.ravel()

    # Compute the FFT.
    fourier = np.fft.rfft(img_vector)
    
    # Record the top 'k' coefficients.
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier))
    coefficients = fourier[sorted_indices[:k]]

    """
    # Zero out the rest and reconstruct.
    sparse_fourier = fourier
    sparse_fourier[sorted_indices[k:]] = 0
    return np.fft.irfft(sparse_fourier)
    """
    
    # Generate basis matrix for these indices.
    basis = np.zeros((len(img_vector), k))
    for i in range(k):

        # Set up a dummy vector with only one index high.
        dummy_vector = np.zeros(len(fourier))
        dummy_vector[sorted_indices[i]] = 1

        # Take the IFFT.
        basis_vector = np.fft.irfft(dummy_vector)

        # Append to basis matrix.
        basis[:, i] = basis_vector

    return basis, coefficients

def basisSketchL1(img, alpha, basis_oversampling=1.0):
    """
    Sketch the image. Procedure: 
    1. Choose a random basis.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||Ax - y||_2^2 + alpha * ||x||_1 : y = image, x = representation, A = basis
    """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(img.ravel()).T
    
    # Generate a random basis.
    basis = np.random.randn(len(img_vector),
                            int(len(img_vector) * basis_oversampling))

    # Construct the problem.
    coefficients = cvx.Variable(basis.shape[1])
    L2 = cvx.norm(basis * coefficients - img_vector, 2)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2**2) + alpha*cvx.Minimize(L1)
    problem = cvx.Problem(objective)

    # Solve.
    problem.solve(verbose=True, solver='CVXOPT')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return basis, coefficients.value
