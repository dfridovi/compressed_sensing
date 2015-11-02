"""
Test script. Run Fourier decomposition on a 1D signal and see what we get.
"""

import numpy as np
import matplotlib.pyplot as plt
import Sketching as sketch

# Parameters.
LENGTH = 1000
K = 1

# Generate a random 1D signal.
signal = np.random.randn(LENGTH)

# Obtain Fourier basis.
basis, coefficients = sketch.basisFourier(signal, K)

# Compute reconstruction.
reconstruction = np.zeros(LENGTH)
for i in range(K):
    component = basis[:,i] * coefficients[i]
    reconstruction += np.real(component)

# Plot.
plt.plot(signal, 'b')
plt.plot(reconstruction, 'r-')
plt.title("Reconstructing a random signal with %d Fourier basis vectors." % K)
plt.show()
