"""Part 1.1 - 2D convolution from scratch (our own implementation).

Conventions used everywhere in this project
-------------------------------------------
* Images are 2D float arrays (H, W). Color images are handled channel by channel
  by the caller.
* Filters ("kernels") are 2D arrays with ODD height and ODD width, so they have a
  well-defined center pixel.
* Output has the same shape as the input ("same" size) and is float64.
* True convolution: the kernel is flipped along both axes, then slid over the image
  and multiplied/summed (correlation with the flipped kernel).
"""
import numpy as np


def _check_inputs(image, kernel):
    """Convert to float64 arrays and validate shapes."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D (grayscale), got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kernel.shape}")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"kernel height and width must be odd, got {kernel.shape}")
    return image, kernel


def pad_image(image, ph, pw, pad_mode="zeros"):
    """Pad `ph` rows on top/bottom and `pw` columns on left/right.

    pad_mode="zeros"   -> Part 1 (as the assignment asks)
    pad_mode="reflect" -> Parts 2 and 3 (np.pad(..., mode="reflect"))
    """
    widths = ((ph, ph), (pw, pw))
    if pad_mode == "zeros":
        return np.pad(image, widths, mode="constant", constant_values=0.0)
    if pad_mode == "reflect":
        return np.pad(image, widths, mode="reflect")
    raise ValueError(f"unknown pad_mode: {pad_mode!r}")


def conv2d_loop(image, kernel, pad_mode="zeros"):
    """2D convolution with explicit loops over every pixel and every filter entry.

    out[i, j] = sum_{u, v} flipped[u, v] * padded[i + u, j + v]
    where flipped = kernel flipped along both axes.
    """
    image, kernel = _check_inputs(image, kernel)
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2            # how far the filter reaches from its center

    padded = pad_image(image, ph, pw, pad_mode)   # shape (H + 2*ph, W + 2*pw)
    flipped = kernel[::-1, ::-1]                   # the "convolution" flip

    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):                  # loop over output pixels (rows)
        for j in range(W):              # loop over output pixels (columns)
            total = 0.0
            for u in range(kh):         # loop over filter entries (rows)
                for v in range(kw):     # loop over filter entries (columns)
                    total += flipped[u, v] * padded[i + u, j + v]
            out[i, j] = total
    return out


def conv2d_vectorized(image, kernel, pad_mode="zeros"):
    """2D convolution with NO loop over pixels: one whole-image operation per filter entry.

    Each filter entry (u, v) contributes  flipped[u, v] * (image shifted by (u, v)).
    The shifted image is just the slice padded[u:u+H, v:v+W] (a view, no copying).
    """
    image, kernel = _check_inputs(image, kernel)
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = pad_image(image, ph, pw, pad_mode)
    flipped = kernel[::-1, ::-1]

    out = np.zeros((H, W), dtype=np.float64)
    for u in range(kh):                  # loop over filter entries only
        for v in range(kw):
            out += flipped[u, v] * padded[u:u + H, v:v + W]
    return out


def separable_factors(kernel, tol=1e-10):
    """SVD rank-1 test.  Returns (col, row) with kernel ~= col @ row, or None if not rank 1.

    K = U diag(s) Vt.  K has rank 1 (up to rounding) when the second singular value is
    negligible compared to the first:  s[1] <= tol * s[0].
    col has shape (kh, 1) (vertical filter), row has shape (1, kw) (horizontal filter).
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    U, s, Vt = np.linalg.svd(kernel)
    if s[0] == 0.0:                                   # all-zero kernel: trivially separable
        return np.zeros((kernel.shape[0], 1)), np.zeros((1, kernel.shape[1]))
    if len(s) > 1 and s[1] > tol * s[0]:
        return None
    scale = np.sqrt(s[0])                             # split the weight evenly between the two
    col = (U[:, 0] * scale).reshape(-1, 1)
    row = (Vt[0, :] * scale).reshape(1, -1)
    return col, row


def conv2d_separable(image, kernel, pad_mode="zeros", tol=1e-10):
    """Convolution that exploits separability when the kernel has rank 1.

    rank 1     : convolve with the column, then with the row   (kh + kw multiply-adds / pixel)
    otherwise  : fall back to conv2d_vectorized                 (kh * kw multiply-adds / pixel)
    """
    image, kernel = _check_inputs(image, kernel)
    factors = separable_factors(kernel, tol)
    if factors is None:
        return conv2d_vectorized(image, kernel, pad_mode)
    col, row = factors
    tmp = conv2d_vectorized(image, col, pad_mode)     # vertical pass
    return conv2d_vectorized(tmp, row, pad_mode)      # horizontal pass
