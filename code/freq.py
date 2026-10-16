"""Part 2.1 / 2.2 - Gaussian low-pass / high-pass filtering, hybrid images, Fourier view.

All filtering uses OUR convolution (conv2d_separable) with reflect padding (Part 2 rule).
The Fourier transform is the one library call the assignment allows here.
"""
import numpy as np
from conv import conv2d_separable
from filters import gaussian_kernel, default_size


def gaussian_blur(image, sigma):
    """Gaussian blur, reflect padding. Works on (H, W) or (H, W, C) float images (per channel)."""
    kernel = gaussian_kernel(default_size(sigma), sigma)          # (2*ceil(3 sigma)+1)^2, sums to 1
    if image.ndim == 2:
        return conv2d_separable(image, kernel, pad_mode="reflect")
    return np.stack([conv2d_separable(image[..., c], kernel, pad_mode="reflect")
                     for c in range(image.shape[2])], axis=-1)


def low_pass(image, sigma):
    return gaussian_blur(image, sigma)


def high_pass(image, sigma):
    """image minus its blur: keeps only the fine detail; values go negative."""
    return np.asarray(image, dtype=np.float64) - gaussian_blur(image, sigma)


def hybrid_image(im_low, im_high, sigma_low, sigma_high):
    """Low frequencies of im_low (cutoff set by sigma_low) + high frequencies of im_high."""
    return low_pass(im_low, sigma_low) + high_pass(im_high, sigma_high)


def to_gray(rgb):
    """Luminance (ITU-R BT.601 weights); an allowed color conversion."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb if rgb.ndim == 2 else rgb @ np.array([0.299, 0.587, 0.114])


def log_fft_magnitude(gray, eps=1e-8):
    """The assignment's formula  np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray)))).
    eps only guards against log(0); it does not change the picture."""
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + eps)


def gaussian_cutoff_frequency(sigma):
    """Frequency (cycles/pixel) where a Gaussian low-pass of this sigma has amplitude 1/2."""
    return np.sqrt(2.0 * np.log(2.0)) / (2.0 * np.pi * sigma)


def downsample(image, factor):
    """Keep every `factor`-th row and column (no filtering at all)."""
    return image[::factor, ::factor]


def blur_then_downsample(image, factor, sigma):
    """Gaussian prefilter (our convolution, reflect padding), then keep every `factor`-th pixel."""
    return downsample(gaussian_blur(image, sigma), factor)
