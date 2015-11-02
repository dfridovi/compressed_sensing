"""
A set of functions for sketching (i.e. randomly compressing to low-rank) an image.
"""

import numpy as np

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
