"""Part 2.3 - Gaussian and Laplacian pyramids, written from scratch.

No cv2.pyrDown / cv2.pyrUp / skimage pyramid functions.  All blurring uses OUR convolution
(conv2d_separable) with reflect padding, and the specified kernel  1/16 * [1 4 6 4 1]  along x and y.
Images are (H, W) or (H, W, C) float arrays; color is handled channel by channel.
"""
import numpy as np
from conv import conv2d_separable

KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0       # binomial, sigma = 1, sums to 1
KERNEL_2D = np.outer(KERNEL_1D, KERNEL_1D)                    # the same filter along x and y (rank 1)


def _blur(image):
    """Blur with KERNEL_2D, reflect padding (every channel separately)."""
    if image.ndim == 2:
        return conv2d_separable(image, KERNEL_2D, pad_mode="reflect")
    return np.stack([conv2d_separable(image[..., c], KERNEL_2D, pad_mode="reflect")
                     for c in range(image.shape[2])], axis=-1)


def reduce(image):
    """Blur, then keep every other row and column.  (H, W) -> (ceil(H/2), ceil(W/2))."""
    image = np.asarray(image, dtype=np.float64)
    return _blur(image)[::2, ::2]


def expand(image, out_shape=None):
    """Put each pixel at the top-left of a 2x2 block of zeros, blur, multiply by 4.

    (h, w) -> (2h, 2w).  If `out_shape` is given it must be (2h, 2w) or (2h-1, 2w-1)-style: when the
    level we are rebuilding had an ODD side, the expansion has one row/column too many, and the last
    one is dropped.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]
    up = np.zeros((2 * h, 2 * w) + image.shape[2:], dtype=np.float64)
    up[::2, ::2] = image                                       # 3 pixels in 4 are zeros ...
    out = 4.0 * _blur(up)                                      # ... so multiply by 4 to keep the brightness
    if out_shape is not None:
        th, tw = out_shape[:2]
        if th not in (2 * h, 2 * h - 1) or tw not in (2 * w, 2 * w - 1):
            raise ValueError(f"cannot expand {(h, w)} to {(th, tw)}")
        out = out[:th, :tw]                                    # drop the extra last row / column
    return out


def max_levels(shape):
    """Number of levels until the smaller side reaches 1 pixel."""
    n, side = 1, min(shape[:2])
    while side > 1:
        side = (side + 1) // 2
        n += 1
    return n


def gaussian_pyramid(image, levels):
    """[G0, G1, ...]: G0 is the image, each next level is reduce() of the one before."""
    pyr = [np.asarray(image, dtype=np.float64)]
    for _ in range(levels - 1):
        pyr.append(reduce(pyr[-1]))
    return pyr


def laplacian_pyramid(gaussian):
    """[L0, L1, ..., G_last]: L_i = G_i - expand(G_{i+1}) at G_i's size; the last entry is the smallest Gaussian level."""
    lap = [g - expand(g_next, g.shape) for g, g_next in zip(gaussian[:-1], gaussian[1:])]
    lap.append(gaussian[-1].copy())
    return lap


def reconstruct(laplacian):
    """Start from the smallest level; repeat: expand, then add the Laplacian level of that size."""
    image = laplacian[-1]
    for lap in reversed(laplacian[:-1]):
        image = lap + expand(image, lap.shape)
    return image


def lecture_style_pyramid(gaussian):
    """For comparison only: 'each level minus its own blur' (no expand).  L'_i = G_i - blur(G_i)."""
    return [g - _blur(g) for g in gaussian]
