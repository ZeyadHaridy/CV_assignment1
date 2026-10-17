"""Part 1.2 / 1.3 - finite differences, derivative-of-Gaussian, gradient magnitude, edges.

All filtering goes through OUR convolution (conv2d_separable, which falls back to the
vectorized version when a kernel is not rank 1).  Part 1 uses zero padding, as the
assignment asks; pad_mode can be changed for comparison.
"""
import numpy as np
from conv import conv2d_separable
from filters import DX, DY, gaussian_kernel, default_size


def gradient_from_filters(image, kx, ky, pad_mode="zeros"):
    """Filter the image with an x-filter and a y-filter; return (gx, gy, magnitude)."""
    gx = conv2d_separable(image, kx, pad_mode)
    gy = conv2d_separable(image, ky, pad_mode)
    return gx, gy, np.sqrt(gx ** 2 + gy ** 2)


def finite_difference_gradient(image, pad_mode="zeros"):
    """1.2: derivatives with Dx = [1 0 -1] and Dy = Dx^T."""
    return gradient_from_filters(image, DX, DY, pad_mode)


def dog_filters(sigma, size=None, pad_mode="zeros"):
    """1.3: filter the Gaussian kernel itself with Dx and Dy -> the two DoG filters.

    Same-size output (as in 1.1), so a size x size Gaussian gives size x size DoG filters.
    """
    size = default_size(sigma) if size is None else size
    G = gaussian_kernel(size, sigma)
    return conv2d_separable(G, DX, pad_mode), conv2d_separable(G, DY, pad_mode)


def dog_gradient(image, sigma, size=None, pad_mode="zeros"):
    """1.3: one convolution per direction with the DoG filters; returns (gx, gy, magnitude)."""
    dog_x, dog_y = dog_filters(sigma, size)
    return gradient_from_filters(image, dog_x, dog_y, pad_mode)


def threshold_edges(magnitude, threshold):
    """Binary edge image: True where the gradient magnitude exceeds the threshold."""
    return magnitude > threshold


# ---------------------------------------------------------------- display helpers
# These only build a COPY for display; the underlying float arrays are never changed.
def signed_for_display(d, scale=None):
    """Map a signed image to [0, 1] with ZERO at mid-gray (0.5): 0.5 + d / (2*scale)."""
    scale = np.abs(d).max() if scale is None else scale
    return np.clip(0.5 + 0.5 * d / scale, 0.0, 1.0)


def magnitude_for_display(m, scale=None):
    """Map a non-negative image to [0, 1] by dividing by `scale` (default: its maximum)."""
    scale = m.max() if scale is None else scale
    return np.clip(m / scale, 0.0, 1.0)
