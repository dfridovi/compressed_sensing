"""
Test script. Run sketching with L1 penalization on a 1D signal.
"""

import numpy as np
import matplotlib.pyplot as plt
import Sketching as sketch

# Parameters.
LENGTH = 1000
K = 100
ALPHA = 10.0

# Generate a random 1D signal.
signal = np.random.randn(LENGTH)

# Obtain Fourier basis.
basis, coefficients = sketch.basisSketchL1(signal, K, ALPHA)

# Compute reconstruction.
reconstruction = basis * coefficients

# Plot.
plt.plot(signal, 'b')
plt.plot(reconstruction, 'r-')
plt.title("Reconstructing a random signal with %d random basis vectors." % K)
plt.show()
