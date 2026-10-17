"""Part 2.4 - multiresolution blending (Burt & Adelson), built on OUR pyramids from 2.3."""
import numpy as np
from pyramids import gaussian_pyramid, laplacian_pyramid, reconstruct


def _as_mask(mask, image):
    """Mask as float64 in [0, 1] with shape (H, W); must match the image's height and width."""
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 3:                       # an RGB mask image: use one channel
        mask = mask[..., 0]
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} and image {image.shape[:2]} must have the same height and width")
    return mask


def hard_cut(im1, im2, mask):
    """Plain paste along the mask: M*I1 + (1-M)*I2."""
    im1, im2 = np.asarray(im1, dtype=np.float64), np.asarray(im2, dtype=np.float64)
    m = _as_mask(mask, im1)
    m = m[..., None] if im1.ndim == 3 else m
    return m * im1 + (1.0 - m) * im2


def multires_blend(im1, im2, mask, levels, return_parts=False):
    """Blend im1 (where mask = 1) and im2 (where mask = 0), band by band.

      L1, L2 : Laplacian pyramids of the two images         (levels entries each; last = smallest Gaussian level)
      GM     : Gaussian pyramid of the (sharp) mask
      out_i  = GM_i * L1_i + (1 - GM_i) * L2_i              at every level, then rebuild as in 2.3
    """
    im1, im2 = np.asarray(im1, dtype=np.float64), np.asarray(im2, dtype=np.float64)
    if im1.shape != im2.shape:
        raise ValueError(f"images must have the same shape, got {im1.shape} and {im2.shape}")
    m = _as_mask(mask, im1)
    L1 = laplacian_pyramid(gaussian_pyramid(im1, levels))
    L2 = laplacian_pyramid(gaussian_pyramid(im2, levels))
    GM = gaussian_pyramid(m, levels)
    expand_m = (lambda g: g[..., None]) if im1.ndim == 3 else (lambda g: g)
    A = [expand_m(gm) * a for gm, a in zip(GM, L1)]            # M * L1
    B = [(1.0 - expand_m(gm)) * b for gm, b in zip(GM, L2)]    # (1 - M) * L2
    combined = [a + b for a, b in zip(A, B)]
    out = reconstruct(combined)
    if return_parts:
        return out, dict(L1=L1, L2=L2, GM=GM, A=A, B=B, combined=combined)
    return out


def effective_alpha(mask, levels):
    """How much of image 1 each pixel receives: blend an all-ones image with an all-zeros image.
    (The blend is linear in the images, so this profile says exactly where and how softly the seam is blended.)"""
    ones = np.ones(mask.shape)
    return multires_blend(ones, np.zeros(mask.shape), mask, levels)
