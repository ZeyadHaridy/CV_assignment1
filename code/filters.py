"""Small helpers that build filters (kernels). The filtering itself is done by conv.py."""
import numpy as np

# Finite-difference filters of 1.2.  Dx is a 1x3 row, Dy = Dx^T is a 3x1 column.
DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T


def gaussian_kernel_1d(size, sigma):
    """Normalized 1D Gaussian with an odd number of taps, centered on the middle tap."""
    if size % 2 == 0:
        raise ValueError("size must be odd")
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel(size, sigma):
    """Normalized 2D Gaussian (size x size, sums to 1) = outer product of two 1D Gaussians."""
    g = gaussian_kernel_1d(size, sigma)
    return np.outer(g, g)


def default_size(sigma):
    """Odd kernel size covering +-3 sigma: 2*ceil(3*sigma)+1."""
    return 2 * int(np.ceil(3 * sigma)) + 1
