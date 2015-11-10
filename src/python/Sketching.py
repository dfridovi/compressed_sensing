"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

import numpy as np
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
    L2 = cvx.sum_squares(basis * coefficients - img_vector)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)

    # Solve.
    problem.solve(verbose=True, solver='SCS')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return basis, coefficients.value

def basisSketchFourierL1(img, alpha, basis_oversampling=1.0):
    """
    Sketch the image in the Fourier domain. Procedure: 
    1. Choose a random matrix to mix the Fourier components.
    2. Solve the L1-penalized least-squares problem to obtain the representation.
    
    min_x ||FAx - y||_2^2 + alpha * ||x||_1, where y = image, 
                                                   x = representation, 
                                                   A = mixing matrix,
                                                   F = Fourier basis
    """

    # Unravel this image into a single column vector.
    img_vector = np.asmatrix(img.ravel()).T
    
    # Generate a random mixing matrix.
    mixing_matrix = np.random.randn(len(img_vector),
                                    int(len(img_vector) * basis_oversampling))

    # TODO!! Fix it so the cast to real only happens at the end.

    # Construct the problem.
    coefficients = cvx.Variable(mixing_matrix.shape[1])
    fourier_basis = np.absolute(computeFourierBasis(len(img_vector)))
    mixed_basis = mixing_matrix * fourier_basis
    fourier_mixing = mixed_basis * coefficients
    L2 = cvx.sum_squares(fourier_mixing - img_vector)
    L1 = cvx.norm(coefficients, 1)
    objective = cvx.Minimize(L2 + alpha*L1)
    constraints = []
    problem = cvx.Problem(objective, constraints)

    # Solve.
    problem.solve(verbose=True, solver='SCS')

    # Print problem status.
    print "Problem status: " + str(problem.status)
    
    return mixing_matrix, coefficients.value
